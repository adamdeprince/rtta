#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <kalman/kalman.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

using InputArray = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using FloatInputArray = nb::ndarray<nb::numpy, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OutputArray = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

enum class InputDType {
    Float32,
    Float64,
};

constexpr long NPY_FLOAT32_TYPE_NUM = 11;
constexpr long NPY_FLOAT64_TYPE_NUM = 12;
constexpr std::size_t SMALL_SCAN_WINDOW_LIMIT = 64;
// Index outputs need absolute offsets; the monotonic queue path is faster than rescanning windows.
constexpr std::size_t SMALL_INDEX_SCAN_WINDOW_LIMIT = 0;

double nan() {
    return std::numeric_limits<double>::quiet_NaN();
}

nb::object make_array(std::vector<double> &&values) {
    auto *storage = new std::vector<double>(std::move(values));
    double *data = storage->data();
    const std::size_t size = storage->size();

    nb::capsule owner(storage, [](void *p) noexcept {
        delete static_cast<std::vector<double> *>(p);
    });

    return nb::cast(OutputArray(data, {size}, owner));
}

nb::arg array_arg(const char *name) {
    nb::arg arg(name);
    arg.noconvert();
    return arg;
}

std::vector<double> make_record_output(nb::handle records) {
    std::vector<double> output;
    const Py_ssize_t size_hint = PyObject_LengthHint(records.ptr(), 0);
    if (size_hint > 0) {
        output.reserve(static_cast<std::size_t>(size_hint));
    }
    return output;
}

double record_value(nb::handle record, const char *field, std::size_t index) {
    if (PyObject_HasAttrString(record.ptr(), field)) {
        PyObject *item = PyObject_GetAttrString(record.ptr(), field);
        if (item != nullptr) {
            nb::object owner = nb::steal<nb::object>(item);
            return nb::cast<double>(owner);
        }
        PyErr_Clear();
    }

    if (PyMapping_Check(record.ptr())) {
        PyObject *item = PyMapping_GetItemString(record.ptr(), field);
        if (item != nullptr) {
            nb::object owner = nb::steal<nb::object>(item);
            return nb::cast<double>(owner);
        }
        PyErr_Clear();
    }

    PyObject *item = PySequence_GetItem(record.ptr(), static_cast<Py_ssize_t>(index));
    if (item != nullptr) {
        nb::object owner = nb::steal<nb::object>(item);
        return nb::cast<double>(owner);
    }
    PyErr_Clear();

    throw nb::type_error("record batch items must expose attributes, mapping keys, or sequence values");
}

double scalar_or_record_value(nb::handle record, const char *field, std::size_t index) {
    const double scalar = PyFloat_AsDouble(record.ptr());
    if (PyErr_Occurred() == nullptr) {
        return scalar;
    }
    PyErr_Clear();
    return record_value(record, field, index);
}

void require_same_size(std::size_t expected, std::size_t actual);

template <typename Indicator, typename Array>
nb::object batch_update1(Indicator &indicator, const Array &input);

template <typename Indicator, typename Array0, typename Array1>
nb::object batch_update2(Indicator &indicator, const Array0 &a, const Array1 &b);

template <typename Indicator, typename Array0, typename Array1, typename Array2>
nb::object batch_update3(Indicator &indicator, const Array0 &a, const Array1 &b, const Array2 &c);

template <typename Indicator, typename Array0, typename Array1, typename Array2, typename Array3>
nb::object batch_update4(
    Indicator &indicator,
    const Array0 &a,
    const Array1 &b,
    const Array2 &c,
    const Array3 &d);

template <typename Indicator, typename Array0, typename Array1, typename Array2, typename Array3, typename Array4>
nb::object batch_update5(
    Indicator &indicator,
    const Array0 &a,
    const Array1 &b,
    const Array2 &c,
    const Array3 &d,
    const Array4 &e);

bool field_uses_close_fallback(const char *field) {
    return std::strcmp(field, "input") == 0 ||
           std::strcmp(field, "value") == 0 ||
           std::strcmp(field, "x") == 0 ||
           std::strcmp(field, "real0") == 0;
}

PyObject *table_get_item(nb::handle table, const char *field) {
    nb::str key(field);
    PyObject *item = PyObject_GetItem(table.ptr(), key.ptr());
    if (item != nullptr) {
        return item;
    }
    PyErr_Clear();

    if (field_uses_close_fallback(field)) {
        nb::str close_key("close");
        item = PyObject_GetItem(table.ptr(), close_key.ptr());
        if (item != nullptr) {
            return item;
        }
        PyErr_Clear();
    }

    return nullptr;
}

bool table_has_column(nb::handle table, const char *field) {
    if (!PyObject_HasAttrString(table.ptr(), "columns")) {
        return false;
    }

    PyObject *item = table_get_item(table, field);
    if (item == nullptr) {
        return false;
    }
    Py_DECREF(item);
    return true;
}

InputDType array_dtype(nb::handle array) {
    PyObject *dtype_object = PyObject_GetAttrString(array.ptr(), "dtype");
    if (dtype_object == nullptr) {
        throw nb::type_error("pandas table columns must expose a NumPy dtype");
    }
    nb::object dtype = nb::steal<nb::object>(dtype_object);

    PyObject *num_object = PyObject_GetAttrString(dtype.ptr(), "num");
    if (num_object == nullptr) {
        throw nb::type_error("pandas table column dtype must expose NumPy dtype.num");
    }
    nb::object num = nb::steal<nb::object>(num_object);

    const long type_num = PyLong_AsLong(num.ptr());
    if (PyErr_Occurred() != nullptr) {
        throw nb::python_error();
    }

    switch (type_num) {
        case NPY_FLOAT32_TYPE_NUM:
            return InputDType::Float32;
        case NPY_FLOAT64_TYPE_NUM:
            return InputDType::Float64;
        default:
            throw nb::type_error("pandas table batch columns must be float32 or float64 so copy=False can stay zero-copy");
    }
}

nb::object table_column_array(nb::handle table, const char *field) {
    PyObject *item = table_get_item(table, field);
    if (item == nullptr) {
        throw nb::type_error("pandas table batch input is missing a required column");
    }

    nb::object column = nb::steal<nb::object>(item);
    nb::object array;
    if (PyObject_HasAttrString(column.ptr(), "to_numpy")) {
        array = column.attr("to_numpy")(nb::arg("copy") = false);
    } else {
        array = column;
    }

    static_cast<void>(array_dtype(array));

    PyObject *ndim_object = PyObject_GetAttrString(array.ptr(), "ndim");
    if (ndim_object == nullptr) {
        throw nb::type_error("pandas table columns must be one-dimensional NumPy arrays");
    }
    nb::object ndim = nb::steal<nb::object>(ndim_object);
    const long ndim_value = PyLong_AsLong(ndim.ptr());
    if (PyErr_Occurred() != nullptr) {
        throw nb::python_error();
    }
    if (ndim_value != 1) {
        throw nb::type_error("pandas table columns must be one-dimensional NumPy arrays");
    }

    PyObject *flags_object = PyObject_GetAttrString(array.ptr(), "flags");
    if (flags_object == nullptr) {
        throw nb::type_error("pandas table columns must expose NumPy array flags");
    }
    nb::object flags = nb::steal<nb::object>(flags_object);
    nb::str contiguous_key("C_CONTIGUOUS");
    PyObject *contiguous_object = PyObject_GetItem(flags.ptr(), contiguous_key.ptr());
    if (contiguous_object == nullptr) {
        PyErr_Clear();
        throw nb::type_error("pandas table columns must expose NumPy contiguity flags");
    }
    nb::object contiguous = nb::steal<nb::object>(contiguous_object);
    const int contiguous_value = PyObject_IsTrue(contiguous.ptr());
    if (contiguous_value < 0) {
        throw nb::python_error();
    }
    if (contiguous_value == 0) {
        throw nb::type_error("pandas table batch columns must be contiguous so they match NumPy array batch speed");
    }

    return array;
}

template <typename Indicator>
nb::object batch_table1(Indicator &indicator, nb::handle table, const char *field) {
    nb::object a = table_column_array(table, field);
    switch (array_dtype(a)) {
        case InputDType::Float32:
            return batch_update1(indicator, nb::cast<FloatInputArray>(a));
        case InputDType::Float64:
            return batch_update1(indicator, nb::cast<InputArray>(a));
    }
    throw nb::type_error("unsupported pandas table column dtype");
}

template <typename Indicator>
nb::object batch_table2(Indicator &indicator, nb::handle table, const char *field0, const char *field1) {
    nb::object a = table_column_array(table, field0);
    nb::object b = table_column_array(table, field1);
    const InputDType dtype = array_dtype(a);
    if (dtype != array_dtype(b)) {
        throw nb::type_error("pandas table batch columns must all use the same floating-point dtype");
    }
    switch (dtype) {
        case InputDType::Float32:
            return batch_update2(indicator, nb::cast<FloatInputArray>(a), nb::cast<FloatInputArray>(b));
        case InputDType::Float64:
            return batch_update2(indicator, nb::cast<InputArray>(a), nb::cast<InputArray>(b));
    }
    throw nb::type_error("unsupported pandas table column dtype");
}

template <typename Indicator>
nb::object batch_table3(
    Indicator &indicator,
    nb::handle table,
    const char *field0,
    const char *field1,
    const char *field2) {
    nb::object a = table_column_array(table, field0);
    nb::object b = table_column_array(table, field1);
    nb::object c = table_column_array(table, field2);
    const InputDType dtype = array_dtype(a);
    if (dtype != array_dtype(b) || dtype != array_dtype(c)) {
        throw nb::type_error("pandas table batch columns must all use the same floating-point dtype");
    }
    switch (dtype) {
        case InputDType::Float32:
            return batch_update3(
                indicator,
                nb::cast<FloatInputArray>(a),
                nb::cast<FloatInputArray>(b),
                nb::cast<FloatInputArray>(c));
        case InputDType::Float64:
            return batch_update3(indicator, nb::cast<InputArray>(a), nb::cast<InputArray>(b), nb::cast<InputArray>(c));
    }
    throw nb::type_error("unsupported pandas table column dtype");
}

template <typename Indicator>
nb::object batch_table4(
    Indicator &indicator,
    nb::handle table,
    const char *field0,
    const char *field1,
    const char *field2,
    const char *field3) {
    nb::object a = table_column_array(table, field0);
    nb::object b = table_column_array(table, field1);
    nb::object c = table_column_array(table, field2);
    nb::object d = table_column_array(table, field3);
    const InputDType dtype = array_dtype(a);
    if (dtype != array_dtype(b) || dtype != array_dtype(c) || dtype != array_dtype(d)) {
        throw nb::type_error("pandas table batch columns must all use the same floating-point dtype");
    }
    switch (dtype) {
        case InputDType::Float32:
            return batch_update4(
                indicator,
                nb::cast<FloatInputArray>(a),
                nb::cast<FloatInputArray>(b),
                nb::cast<FloatInputArray>(c),
                nb::cast<FloatInputArray>(d));
        case InputDType::Float64:
            return batch_update4(
                indicator,
                nb::cast<InputArray>(a),
                nb::cast<InputArray>(b),
                nb::cast<InputArray>(c),
                nb::cast<InputArray>(d));
    }
    throw nb::type_error("unsupported pandas table column dtype");
}

template <typename Indicator, typename Callback>
auto dispatch_table1(Indicator &indicator, nb::handle table, const char *field0, Callback callback) {
    nb::object a = table_column_array(table, field0);
    switch (array_dtype(a)) {
        case InputDType::Float32:
            return callback(indicator, nb::cast<FloatInputArray>(a));
        case InputDType::Float64:
            return callback(indicator, nb::cast<InputArray>(a));
    }
    throw nb::type_error("unsupported pandas table column dtype");
}

template <typename Indicator, typename Callback>
auto dispatch_table2(Indicator &indicator, nb::handle table, const char *field0, const char *field1, Callback callback) {
    nb::object a = table_column_array(table, field0);
    nb::object b = table_column_array(table, field1);
    const InputDType dtype = array_dtype(a);
    if (dtype != array_dtype(b)) {
        throw nb::type_error("pandas table batch columns must all use the same floating-point dtype");
    }
    switch (dtype) {
        case InputDType::Float32:
            return callback(indicator, nb::cast<FloatInputArray>(a), nb::cast<FloatInputArray>(b));
        case InputDType::Float64:
            return callback(indicator, nb::cast<InputArray>(a), nb::cast<InputArray>(b));
    }
    throw nb::type_error("unsupported pandas table column dtype");
}

template <typename Indicator, typename Callback>
auto dispatch_table3(
    Indicator &indicator,
    nb::handle table,
    const char *field0,
    const char *field1,
    const char *field2,
    Callback callback) {
    nb::object a = table_column_array(table, field0);
    nb::object b = table_column_array(table, field1);
    nb::object c = table_column_array(table, field2);
    const InputDType dtype = array_dtype(a);
    if (dtype != array_dtype(b) || dtype != array_dtype(c)) {
        throw nb::type_error("pandas table batch columns must all use the same floating-point dtype");
    }
    switch (dtype) {
        case InputDType::Float32:
            return callback(indicator, nb::cast<FloatInputArray>(a), nb::cast<FloatInputArray>(b), nb::cast<FloatInputArray>(c));
        case InputDType::Float64:
            return callback(indicator, nb::cast<InputArray>(a), nb::cast<InputArray>(b), nb::cast<InputArray>(c));
    }
    throw nb::type_error("unsupported pandas table column dtype");
}

template <typename Indicator, typename Callback>
auto dispatch_table4(
    Indicator &indicator,
    nb::handle table,
    const char *field0,
    const char *field1,
    const char *field2,
    const char *field3,
    Callback callback) {
    nb::object a = table_column_array(table, field0);
    nb::object b = table_column_array(table, field1);
    nb::object c = table_column_array(table, field2);
    nb::object d = table_column_array(table, field3);
    const InputDType dtype = array_dtype(a);
    if (dtype != array_dtype(b) || dtype != array_dtype(c) || dtype != array_dtype(d)) {
        throw nb::type_error("pandas table batch columns must all use the same floating-point dtype");
    }
    switch (dtype) {
        case InputDType::Float32:
            return callback(
                indicator,
                nb::cast<FloatInputArray>(a),
                nb::cast<FloatInputArray>(b),
                nb::cast<FloatInputArray>(c),
                nb::cast<FloatInputArray>(d));
        case InputDType::Float64:
            return callback(
                indicator,
                nb::cast<InputArray>(a),
                nb::cast<InputArray>(b),
                nb::cast<InputArray>(c),
                nb::cast<InputArray>(d));
    }
    throw nb::type_error("unsupported pandas table column dtype");
}

template <typename Indicator, typename Callback>
auto dispatch_table5(
    Indicator &indicator,
    nb::handle table,
    const char *field0,
    const char *field1,
    const char *field2,
    const char *field3,
    const char *field4,
    Callback callback) {
    nb::object a = table_column_array(table, field0);
    nb::object b = table_column_array(table, field1);
    nb::object c = table_column_array(table, field2);
    nb::object d = table_column_array(table, field3);
    nb::object e = table_column_array(table, field4);
    const InputDType dtype = array_dtype(a);
    if (dtype != array_dtype(b) || dtype != array_dtype(c) || dtype != array_dtype(d) || dtype != array_dtype(e)) {
        throw nb::type_error("pandas table batch columns must all use the same floating-point dtype");
    }
    switch (dtype) {
        case InputDType::Float32:
            return callback(
                indicator,
                nb::cast<FloatInputArray>(a),
                nb::cast<FloatInputArray>(b),
                nb::cast<FloatInputArray>(c),
                nb::cast<FloatInputArray>(d),
                nb::cast<FloatInputArray>(e));
        case InputDType::Float64:
            return callback(
                indicator,
                nb::cast<InputArray>(a),
                nb::cast<InputArray>(b),
                nb::cast<InputArray>(c),
                nb::cast<InputArray>(d),
                nb::cast<InputArray>(e));
    }
    throw nb::type_error("unsupported pandas table column dtype");
}

template <typename Indicator>
nb::object batch_records_one(Indicator &indicator, nb::iterable records, const char *field) {
    if (table_has_column(records, field)) {
        return batch_table1(indicator, records, field);
    }

    std::vector<double> output = make_record_output(records);
    for (nb::handle record : records) {
        output.push_back(indicator.update(record_value(record, field, 0)));
    }
    return make_array(std::move(output));
}

template <typename Indicator>
nb::object batch_records_two(Indicator &indicator, nb::iterable records, const char *field0, const char *field1) {
    if (table_has_column(records, field0)) {
        return batch_table2(indicator, records, field0, field1);
    }

    std::vector<double> output = make_record_output(records);
    for (nb::handle record : records) {
        output.push_back(indicator.update(record_value(record, field0, 0), record_value(record, field1, 1)));
    }
    return make_array(std::move(output));
}

template <typename Indicator>
nb::object batch_records_three(
    Indicator &indicator,
    nb::iterable records,
    const char *field0,
    const char *field1,
    const char *field2) {
    if (table_has_column(records, field0)) {
        return batch_table3(indicator, records, field0, field1, field2);
    }

    std::vector<double> output = make_record_output(records);
    for (nb::handle record : records) {
        output.push_back(indicator.update(
            record_value(record, field0, 0),
            record_value(record, field1, 1),
            record_value(record, field2, 2)));
    }
    return make_array(std::move(output));
}

template <typename Indicator>
nb::object batch_records_four(
    Indicator &indicator,
    nb::iterable records,
    const char *field0,
    const char *field1,
    const char *field2,
    const char *field3) {
    if (table_has_column(records, field0)) {
        return batch_table4(indicator, records, field0, field1, field2, field3);
    }

    std::vector<double> output = make_record_output(records);
    for (nb::handle record : records) {
        output.push_back(indicator.update(
            record_value(record, field0, 0),
            record_value(record, field1, 1),
            record_value(record, field2, 2),
            record_value(record, field3, 3)));
    }
    return make_array(std::move(output));
}

template <typename Indicator>
nb::object batch_records_five(
    Indicator &indicator,
    nb::iterable records,
    const char *field0,
    const char *field1,
    const char *field2,
    const char *field3,
    const char *field4) {
    if (table_has_column(records, field0)) {
        return dispatch_table5(
            indicator,
            records,
            field0,
            field1,
            field2,
            field3,
            field4,
            [](auto &self, const auto &a, const auto &b, const auto &c, const auto &d, const auto &e) {
                return batch_update5(self, a, b, c, d, e);
            });
    }

    std::vector<double> output = make_record_output(records);
    for (nb::handle record : records) {
        output.push_back(indicator.update(
            record_value(record, field0, 0),
            record_value(record, field1, 1),
            record_value(record, field2, 2),
            record_value(record, field3, 3),
            record_value(record, field4, 4)));
    }
    return make_array(std::move(output));
}

void require_same_size(std::size_t expected, std::size_t actual) {
    if (expected != actual) {
        throw nb::value_error("batch input arrays must have the same length");
    }
}

template <typename Indicator, typename Array>
nb::object batch_update1(Indicator &indicator, const Array &input) {
    const std::size_t size = input.shape(0);
    std::vector<double> output(size);
    const auto *values = input.data();
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = indicator.update(static_cast<double>(values[i]));
    }
    return make_array(std::move(output));
}

template <typename Indicator, typename Array0, typename Array1>
nb::object batch_update2(Indicator &indicator, const Array0 &a, const Array1 &b) {
    const std::size_t size = a.shape(0);
    require_same_size(size, b.shape(0));
    std::vector<double> output(size);
    const auto *a_values = a.data();
    const auto *b_values = b.data();
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = indicator.update(static_cast<double>(a_values[i]), static_cast<double>(b_values[i]));
    }
    return make_array(std::move(output));
}

template <typename Indicator, typename Array0, typename Array1, typename Array2>
nb::object batch_update3(Indicator &indicator, const Array0 &a, const Array1 &b, const Array2 &c) {
    const std::size_t size = a.shape(0);
    require_same_size(size, b.shape(0));
    require_same_size(size, c.shape(0));
    std::vector<double> output(size);
    const auto *a_values = a.data();
    const auto *b_values = b.data();
    const auto *c_values = c.data();
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = indicator.update(
            static_cast<double>(a_values[i]),
            static_cast<double>(b_values[i]),
            static_cast<double>(c_values[i]));
    }
    return make_array(std::move(output));
}

template <typename Indicator, typename Array0, typename Array1, typename Array2, typename Array3>
nb::object batch_update4(
    Indicator &indicator,
    const Array0 &a,
    const Array1 &b,
    const Array2 &c,
    const Array3 &d) {
    const std::size_t size = a.shape(0);
    require_same_size(size, b.shape(0));
    require_same_size(size, c.shape(0));
    require_same_size(size, d.shape(0));
    std::vector<double> output(size);
    const auto *a_values = a.data();
    const auto *b_values = b.data();
    const auto *c_values = c.data();
    const auto *d_values = d.data();
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = indicator.update(
            static_cast<double>(a_values[i]),
            static_cast<double>(b_values[i]),
            static_cast<double>(c_values[i]),
            static_cast<double>(d_values[i]));
    }
    return make_array(std::move(output));
}

template <typename Indicator, typename Array0, typename Array1, typename Array2, typename Array3, typename Array4>
nb::object batch_update5(
    Indicator &indicator,
    const Array0 &a,
    const Array1 &b,
    const Array2 &c,
    const Array3 &d,
    const Array4 &e) {
    const std::size_t size = a.shape(0);
    require_same_size(size, b.shape(0));
    require_same_size(size, c.shape(0));
    require_same_size(size, d.shape(0));
    require_same_size(size, e.shape(0));
    std::vector<double> output(size);
    const auto *a_values = a.data();
    const auto *b_values = b.data();
    const auto *c_values = c.data();
    const auto *d_values = d.data();
    const auto *e_values = e.data();
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = indicator.update(
            static_cast<double>(a_values[i]),
            static_cast<double>(b_values[i]),
            static_cast<double>(c_values[i]),
            static_cast<double>(d_values[i]),
            static_cast<double>(e_values[i]));
    }
    return make_array(std::move(output));
}

struct EaseOfMovementResult {
    double ease_of_movement;
    double sma;
};

struct EaseOfMovementBatchResult {
    nb::object ease_of_movement;
    nb::object sma;
};

struct LinearRegressionResult {
    double value;
    double slope;
    double intercept;
    double angle;
    double tsf;
};

struct LinearRegressionBatchResult {
    nb::object value;
    nb::object slope;
    nb::object intercept;
    nb::object angle;
    nb::object tsf;
};

struct RollingMinMaxResult {
    double min;
    double max;
    double min_index;
    double max_index;
};

struct HighLowResult {
    double min;
    double max;
};

struct HighLowBatchResult {
    nb::object min;
    nb::object max;
};

struct HighLowIndexResult {
    double min_index;
    double max_index;
};

struct HighLowIndexBatchResult {
    nb::object min_index;
    nb::object max_index;
};

struct KeltnerChannelResult {
    double middle;
    double upper;
    double lower;
};

struct KeltnerChannelBatchResult {
    nb::object middle;
    nb::object upper;
    nb::object lower;
};

struct SuperTrendResult {
    double value;
    double direction;
    double upper;
    double lower;
};

struct SuperTrendBatchResult {
    nb::object value;
    nb::object direction;
    nb::object upper;
    nb::object lower;
};

struct DonchianChannelResult {
    double upper;
    double lower;
    double middle;
    double width;
    double percent;
};

struct DonchianChannelBatchResult {
    nb::object upper;
    nb::object lower;
    nb::object middle;
    nb::object width;
    nb::object percent;
};

struct FibonacciRetracementLevelsResult {
    double level0;
    double level236;
    double level382;
    double level500;
    double level618;
    double level100;
};

struct FibonacciRetracementLevelsBatchResult {
    nb::object level0;
    nb::object level236;
    nb::object level382;
    nb::object level500;
    nb::object level618;
    nb::object level100;
};

struct AroonResult {
    double down;
    double up;
};

struct AroonBatchResult {
    nb::object down;
    nb::object up;
};

struct VortexResult {
    double positive;
    double negative;
    double difference;
};

struct VortexBatchResult {
    nb::object positive;
    nb::object negative;
    nb::object difference;
};

struct KSTOscillatorResult {
    double kst;
    double signal;
    double difference;
};

struct KSTOscillatorBatchResult {
    nb::object kst;
    nb::object signal;
    nb::object difference;
};

struct IchimokuResult {
    double conversion;
    double base;
    double span_a;
    double span_b;
};

struct IchimokuBatchResult {
    nb::object conversion;
    nb::object base;
    nb::object span_a;
    nb::object span_b;
};

struct PercentagePriceResult {
    double ppo;
    double signal;
    double histogram;
};

struct PercentagePriceBatchResult {
    nb::object ppo;
    nb::object signal;
    nb::object histogram;
};

struct PercentageVolumeResult {
    double pvo;
    double signal;
    double histogram;
};

struct PercentageVolumeBatchResult {
    nb::object pvo;
    nb::object signal;
    nb::object histogram;
};

struct FastStochasticResult {
    double fastk;
    double fastd;
};

struct FastStochasticBatchResult {
    nb::object fastk;
    nb::object fastd;
};

struct StochasticResult {
    double slowk;
    double slowd;
};

struct StochasticBatchResult {
    nb::object slowk;
    nb::object slowd;
};

struct BollingerBandsResult {
    double middle;
    double upper;
    double lower;
};

struct BollingerBandsBatchResult {
    nb::object middle;
    nb::object upper;
    nb::object lower;
};

struct KalmanPredictionBandsResult {
    double middle;
    double upper;
    double lower;
};

struct KalmanPredictionBandsBatchResult {
    nb::object middle;
    nb::object upper;
    nb::object lower;
};

struct KalmanTrendSignalResult {
    double trend;
    double signal;
};

struct KalmanTrendSignalBatchResult {
    nb::object trend;
    nb::object signal;
};

struct KalmanLocalLinearTrendResult {
    double level;
    double trend;
};

struct KalmanLocalLinearTrendBatchResult {
    nb::object level;
    nb::object trend;
};

struct RelativeVigorIndexResult {
    double rvi;
    double signal;
};

struct RelativeVigorIndexBatchResult {
    nb::object rvi;
    nb::object signal;
};

struct KlingerVolumeOscillatorResult {
    double kvo;
    double signal;
    double histogram;
};

struct KlingerVolumeOscillatorBatchResult {
    nb::object kvo;
    nb::object signal;
    nb::object histogram;
};

struct ElderRayIndexResult {
    double bull_power;
    double bear_power;
};

struct ElderRayIndexBatchResult {
    nb::object bull_power;
    nb::object bear_power;
};

struct MesaAdaptiveMovingAverageResult {
    double mama;
    double fama;
};

struct MesaAdaptiveMovingAverageBatchResult {
    nb::object mama;
    nb::object fama;
};

struct KalmanRegressionChannelResult {
    double slope;
    double intercept;
    double middle;
    double upper;
    double lower;
    double spread;
};

struct KalmanRegressionChannelBatchResult {
    nb::object slope;
    nb::object intercept;
    nb::object middle;
    nb::object upper;
    nb::object lower;
    nb::object spread;
};

struct KalmanHedgeRatioResult {
    double hedge_ratio;
    double intercept;
    double spread;
};

struct KalmanHedgeRatioBatchResult {
    nb::object hedge_ratio;
    nb::object intercept;
    nb::object spread;
};

struct TwoFactorKalmanTrendFilterResult {
    double short_trend;
    double long_trend;
    double value;
};

struct TwoFactorKalmanTrendFilterBatchResult {
    nb::object short_trend;
    nb::object long_trend;
    nb::object value;
};

struct KalmanExtremumTrendResult {
    double trend;
    double oscillator;
    double signal;
};

struct KalmanExtremumTrendBatchResult {
    nb::object trend;
    nb::object oscillator;
    nb::object signal;
};

struct AlphaBetaGammaTrackingFilterResult {
    double price;
    double velocity;
    double acceleration;
    double residual;
};

struct AlphaBetaGammaTrackingFilterBatchResult {
    nb::object price;
    nb::object velocity;
    nb::object acceleration;
    nb::object residual;
};

struct InteractingMultipleModelFilterResult {
    double value;
    double velocity;
    double low_vol_probability;
    double high_vol_probability;
    double trend_probability;
    double chop_probability;
};

struct InteractingMultipleModelFilterBatchResult {
    nb::object value;
    nb::object velocity;
    nb::object low_vol_probability;
    nb::object high_vol_probability;
    nb::object trend_probability;
    nb::object chop_probability;
};

struct ParticleFilterTrendResult {
    double trend;
    double velocity;
    double signal;
    double effective_sample_size;
};

struct ParticleFilterTrendBatchResult {
    nb::object trend;
    nb::object velocity;
    nb::object signal;
    nb::object effective_sample_size;
};

struct SavitzkyGolayFilterResult {
    double smooth;
    double first_derivative;
    double second_derivative;
};

struct SavitzkyGolayFilterBatchResult {
    nb::object smooth;
    nb::object first_derivative;
    nb::object second_derivative;
};

struct NadarayaWatsonEnvelopeResult {
    double middle;
    double upper;
    double lower;
};

struct NadarayaWatsonEnvelopeBatchResult {
    nb::object middle;
    nb::object upper;
    nb::object lower;
};

struct GaussianProcessRegressionBandsResult {
    double middle;
    double upper;
    double lower;
};

struct GaussianProcessRegressionBandsBatchResult {
    nb::object middle;
    nb::object upper;
    nb::object lower;
};

struct ZigZagSwingDetectorResult {
    double value;
    double direction;
    double pivot;
    double pivot_index;
};

struct ZigZagSwingDetectorBatchResult {
    nb::object value;
    nb::object direction;
    nb::object pivot;
    nb::object pivot_index;
};

struct RenkoBrickGeneratorResult {
    double brick_open;
    double brick_close;
    double direction;
    double bricks;
    double reversal;
};

struct RenkoBrickGeneratorBatchResult {
    nb::object brick_open;
    nb::object brick_close;
    nb::object direction;
    nb::object bricks;
    nb::object reversal;
};

struct HeikinAshiTransformResult {
    double open;
    double high;
    double low;
    double close;
};

struct HeikinAshiTransformBatchResult {
    nb::object open;
    nb::object high;
    nb::object low;
    nb::object close;
};

struct VolumeProfileResult {
    double point_of_control;
    double value_area_high;
    double value_area_low;
};

struct VolumeProfileBatchResult {
    nb::object point_of_control;
    nb::object value_area_high;
    nb::object value_area_low;
};

struct SpreadFeaturesResult {
    double quoted_spread;
    double effective_spread;
    double realized_spread;
};

struct SpreadFeaturesBatchResult {
    nb::object quoted_spread;
    nb::object effective_spread;
    nb::object realized_spread;
};

inline double result_checksum(double value) {
    return value;
}

inline double result_checksum(const nb::tuple &values) {
    double checksum = 0.0;
    const Py_ssize_t size = PyTuple_GET_SIZE(values.ptr());
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PyTuple_GET_ITEM(values.ptr(), i);
        const double value = PyFloat_AsDouble(item);
        if (PyErr_Occurred() != nullptr) {
            throw nb::python_error();
        }
        checksum += value;
    }
    return checksum;
}

inline double result_checksum(const EaseOfMovementResult &value) {
    return value.ease_of_movement + value.sma;
}

inline double result_checksum(const LinearRegressionResult &value) {
    return value.value + value.slope + value.intercept + value.angle + value.tsf;
}

inline double result_checksum(const RollingMinMaxResult &value) {
    return value.min + value.max;
}

inline double result_checksum(const HighLowResult &value) {
    return value.min + value.max;
}

inline double result_checksum(const HighLowIndexResult &value) {
    return value.min_index + value.max_index;
}

inline double result_checksum(const KeltnerChannelResult &value) {
    return value.middle + value.upper + value.lower;
}

inline double result_checksum(const SuperTrendResult &value) {
    return value.value + value.direction + value.upper + value.lower;
}

inline double result_checksum(const DonchianChannelResult &value) {
    return value.upper + value.lower + value.middle + value.width + value.percent;
}

inline double result_checksum(const FibonacciRetracementLevelsResult &value) {
    return value.level0 + value.level236 + value.level382 + value.level500 + value.level618 + value.level100;
}

inline double result_checksum(const AroonResult &value) {
    return value.down + value.up;
}

inline double result_checksum(const VortexResult &value) {
    return value.positive + value.negative + value.difference;
}

inline double result_checksum(const KSTOscillatorResult &value) {
    return value.kst + value.signal + value.difference;
}

inline double result_checksum(const IchimokuResult &value) {
    return value.conversion + value.base + value.span_a + value.span_b;
}

inline double result_checksum(const PercentagePriceResult &value) {
    return value.ppo + value.signal + value.histogram;
}

inline double result_checksum(const PercentageVolumeResult &value) {
    return value.pvo + value.signal + value.histogram;
}

inline double result_checksum(const FastStochasticResult &value) {
    return value.fastk + value.fastd;
}

inline double result_checksum(const StochasticResult &value) {
    return value.slowk + value.slowd;
}

inline double result_checksum(const BollingerBandsResult &value) {
    return value.middle + value.upper + value.lower;
}

inline double result_checksum(const KalmanPredictionBandsResult &value) {
    return value.middle + value.upper + value.lower;
}

inline double result_checksum(const KalmanTrendSignalResult &value) {
    return value.trend + value.signal;
}

inline double result_checksum(const KalmanLocalLinearTrendResult &value) {
    return value.level + value.trend;
}

inline double result_checksum(const RelativeVigorIndexResult &value) {
    return value.rvi + value.signal;
}

inline double result_checksum(const KlingerVolumeOscillatorResult &value) {
    return value.kvo + value.signal + value.histogram;
}

inline double result_checksum(const ElderRayIndexResult &value) {
    return value.bull_power + value.bear_power;
}

inline double result_checksum(const MesaAdaptiveMovingAverageResult &value) {
    return value.mama + value.fama;
}

inline double result_checksum(const KalmanRegressionChannelResult &value) {
    return value.slope + value.intercept + value.middle + value.upper + value.lower + value.spread;
}

inline double result_checksum(const KalmanHedgeRatioResult &value) {
    return value.hedge_ratio + value.intercept + value.spread;
}

inline double result_checksum(const TwoFactorKalmanTrendFilterResult &value) {
    return value.short_trend + value.long_trend + value.value;
}

inline double result_checksum(const KalmanExtremumTrendResult &value) {
    return value.trend + value.oscillator + value.signal;
}

inline double result_checksum(const AlphaBetaGammaTrackingFilterResult &value) {
    return value.price + value.velocity + value.acceleration + value.residual;
}

inline double result_checksum(const InteractingMultipleModelFilterResult &value) {
    return value.value + value.velocity + value.low_vol_probability + value.high_vol_probability +
           value.trend_probability + value.chop_probability;
}

inline double result_checksum(const ParticleFilterTrendResult &value) {
    return value.trend + value.velocity + value.signal + value.effective_sample_size;
}

inline double result_checksum(const SavitzkyGolayFilterResult &value) {
    return value.smooth + value.first_derivative + value.second_derivative;
}

inline double result_checksum(const NadarayaWatsonEnvelopeResult &value) {
    return value.middle + value.upper + value.lower;
}

inline double result_checksum(const GaussianProcessRegressionBandsResult &value) {
    return value.middle + value.upper + value.lower;
}

inline double result_checksum(const ZigZagSwingDetectorResult &value) {
    return value.value + value.direction + value.pivot + value.pivot_index;
}

inline double result_checksum(const RenkoBrickGeneratorResult &value) {
    return value.brick_open + value.brick_close + value.direction + value.bricks + value.reversal;
}

inline double result_checksum(const HeikinAshiTransformResult &value) {
    return value.open + value.high + value.low + value.close;
}

inline double result_checksum(const VolumeProfileResult &value) {
    return value.point_of_control + value.value_area_high + value.value_area_low;
}

inline double result_checksum(const SpreadFeaturesResult &value) {
    return value.quoted_spread + value.effective_spread + value.realized_spread;
}

class RollingExtremeQueue {
public:
    RollingExtremeQueue(std::size_t capacity, bool maximum)
        : values_(std::max<std::size_t>(capacity, 1)),
          head_(0),
          tail_(0),
          count_(0),
          maximum_(maximum) {}

    inline void push(double value, std::size_t index) {
        if (maximum_) {
            while (!empty() && value >= back().value) {
                pop_back();
            }
        } else {
            while (!empty() && value <= back().value) {
                pop_back();
            }
        }
        push_back({value, index});
    }

    inline void expire_before(std::size_t oldest) {
        while (!empty() && front().index < oldest) {
            pop_front();
        }
    }

    inline double value() const {
        return front().value;
    }

    inline std::size_t index() const {
        return front().index;
    }

    inline void reset() {
        head_ = 0;
        tail_ = 0;
        count_ = 0;
    }

private:
    struct Entry {
        double value;
        std::size_t index;
    };

    inline bool empty() const {
        return count_ == 0;
    }

    inline void advance(std::size_t &index) const {
        ++index;
        if (index == values_.size()) {
            index = 0;
        }
    }

    inline Entry &front() {
        return values_[head_];
    }

    inline const Entry &front() const {
        return values_[head_];
    }

    inline Entry &back() {
        return values_[tail_ == 0 ? values_.size() - 1 : tail_ - 1];
    }

    inline void push_back(Entry entry) {
        if (count_ == values_.size()) {
            pop_front();
        }
        values_[tail_] = entry;
        advance(tail_);
        ++count_;
    }

    inline void pop_front() {
        advance(head_);
        --count_;
    }

    inline void pop_back() {
        tail_ = tail_ == 0 ? values_.size() - 1 : tail_ - 1;
        --count_;
    }

    std::vector<Entry> values_;
    std::size_t head_;
    std::size_t tail_;
    std::size_t count_;
    bool maximum_;
};

class RollingWindow {
public:
    explicit RollingWindow(int window)
        : values_(static_cast<std::size_t>(std::max(window, 1)), 0.0),
          min_(values_.size() + 1, false),
          max_(values_.size() + 1, true),
          next_(0),
          count_(0),
          sequence_(0),
          sum_(0.0) {}

    inline void push(double value) {
        if (count_ == values_.size()) {
            sum_ -= values_[next_];
        } else {
            ++count_;
        }

        const std::size_t index = sequence_++;
        values_[next_] = value;
        next_ = (next_ + 1) % values_.size();
        sum_ += value;

        min_.push(value, index);
        max_.push(value, index);
        expire_old_values();

    }

    inline std::size_t size() const {
        return count_;
    }

    inline std::size_t capacity() const {
        return values_.size();
    }

    inline bool full() const {
        return count_ == values_.size();
    }

    inline double newest() const {
        return at(count_ - 1);
    }

    inline double oldest() const {
        return at(0);
    }

    inline double at(std::size_t oldest_offset) const {
        const std::size_t start = count_ == values_.size() ? next_ : 0;
        return values_[(start + oldest_offset) % values_.size()];
    }

    inline double sum() const {
        return sum_;
    }

    inline double min() const {
        return min_.value();
    }

    inline double max() const {
        return max_.value();
    }

    inline std::size_t min_offset() const {
        return min_.index() - oldest_index();
    }

    inline std::size_t max_offset() const {
        return max_.index() - oldest_index();
    }

private:
    inline std::size_t oldest_index() const {
        return sequence_ - count_;
    }

    inline void expire_old_values() {
        const std::size_t oldest = oldest_index();
        min_.expire_before(oldest);
        max_.expire_before(oldest);
    }

    std::vector<double> values_;
    RollingExtremeQueue min_;
    RollingExtremeQueue max_;
    std::size_t next_;
    std::size_t count_;
    std::size_t sequence_;
    double sum_;
};

class RollingBuffer {
public:
    explicit RollingBuffer(int window)
        : values_(static_cast<std::size_t>(std::max(window, 1)), 0.0),
          next_(0),
          count_(0) {}

    inline double push(double value) {
        double old = 0.0;
        if (full()) {
            old = values_[next_];
        } else {
            ++count_;
        }

        values_[next_] = value;
        next_ = (next_ + 1) % values_.size();
        return old;
    }

    inline std::size_t size() const {
        return count_;
    }

    inline std::size_t capacity() const {
        return values_.size();
    }

    inline bool full() const {
        return count_ == values_.size();
    }

    inline double oldest() const {
        return values_[full() ? next_ : 0];
    }

    inline double at(std::size_t oldest_offset) const {
        const std::size_t start = full() ? next_ : 0;
        return values_[(start + oldest_offset) % values_.size()];
    }

    inline void reset() {
        next_ = 0;
        count_ = 0;
    }

private:
    std::vector<double> values_;
    std::size_t next_;
    std::size_t count_;
};

class RollingSumWindow {
public:
    explicit RollingSumWindow(int window)
        : values_(static_cast<std::size_t>(std::max(window, 1)), 0.0),
          next_(0),
          count_(0),
          sum_(0.0) {}

    inline void push(double value) {
        if (count_ == values_.size()) {
            sum_ -= values_[next_];
        } else {
            ++count_;
        }

        values_[next_] = value;
        next_ = (next_ + 1) % values_.size();
        sum_ += value;
    }

    inline bool full() const {
        return count_ == values_.size();
    }

    inline std::size_t size() const {
        return count_;
    }

    inline double sum() const {
        return sum_;
    }

private:
    std::vector<double> values_;
    std::size_t next_;
    std::size_t count_;
    double sum_;
};

inline void rolling_sum_push(RollingBuffer &buffer, double &sum, double value) {
    if (buffer.full()) {
        sum -= buffer.oldest();
    }
    buffer.push(value);
    sum += value;
}

class RollingExtreme {
public:
    RollingExtreme(int window, bool maximum)
        : window_(static_cast<std::size_t>(std::max(window, 1))),
          count_(0),
          sequence_(0),
          values_(window_ + 1, maximum) {}

    inline void push(double value) {
        if (count_ < window_) {
            ++count_;
        }

        const std::size_t index = sequence_++;
        values_.push(value, index);
        expire_old_values();
    }

    inline bool full() const {
        return count_ == window_;
    }

    inline std::size_t size() const {
        return count_;
    }

    inline double value() const {
        return values_.value();
    }

    inline std::size_t offset() const {
        return values_.index() - oldest_index();
    }

    inline void reset() {
        count_ = 0;
        sequence_ = 0;
        values_.reset();
    }

private:
    inline std::size_t oldest_index() const {
        return sequence_ - count_;
    }

    inline void expire_old_values() {
        const std::size_t oldest = oldest_index();
        values_.expire_before(oldest);
    }

    std::size_t window_;
    std::size_t count_;
    std::size_t sequence_;
    RollingExtremeQueue values_;
};

inline double safe_divide(double numerator, double denominator, double fallback = 0.0) {
    return denominator == 0.0 ? fallback : numerator / denominator;
}

inline double normal_cdf(double value) {
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

inline double true_range(double close, double high, double low, double prev_close) {
    return std::max({
        high - low,
        std::abs(high - prev_close),
        std::abs(low - prev_close),
    });
}

inline double money_flow_multiplier(double close, double high, double low) {
    return high == low ? 0.0 : ((close - low) - (high - close)) / (high - low);
}

namespace batch_kernels {

template <typename Value>
void weighted_moving_average(
    const Value *values,
    std::size_t size,
    int window,
    RollingBuffer &history,
    bool fillna,
    double &weighted_sum,
    double &sum,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const double value = static_cast<double>(values[i]);
        const std::size_t history_size = history.size();
        if (history.full()) {
            const double oldest = history.oldest();
            weighted_sum = weighted_sum - sum + static_cast<double>(window) * value;
            sum += value - oldest;
        } else {
            weighted_sum += static_cast<double>(history_size + 1) * value;
            sum += value;
        }

        history.push(value);
        if (!fillna && !history.full()) {
            output[i] = nan();
        } else {
            const double n = static_cast<double>(history.size());
            output[i] = weighted_sum / (n * (n + 1.0) * 0.5);
        }
    }
}

template <typename Value>
void rolling_variance(
    const Value *values,
    std::size_t size,
    RollingBuffer &history,
    bool fillna,
    double &sum,
    double &sum2,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const double value = static_cast<double>(values[i]);
        if (history.full()) {
            const double oldest = history.oldest();
            sum -= oldest;
            sum2 -= oldest * oldest;
        }

        history.push(value);
        sum += value;
        sum2 += value * value;

        if (!fillna && !history.full()) {
            output[i] = nan();
        } else {
            const double n = static_cast<double>(history.size());
            const double variance = (sum2 - sum * sum / n) / n;
            output[i] = variance < 0.0 && variance > -1e-12 ? 0.0 : variance;
        }
    }
}

template <typename Value>
void rolling_stddev(
    const Value *values,
    std::size_t size,
    RollingBuffer &history,
    bool fillna,
    int window_size,
    long &counter,
    double &sum,
    double &sum2,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const bool return_nan = !fillna && counter < window_size;
        ++counter;

        const double value = static_cast<double>(values[i]);
        if (history.full()) {
            const double oldest = history.oldest();
            sum -= oldest;
            sum2 -= oldest * oldest;
        }

        history.push(value);
        sum += value;
        sum2 += value * value;

        if (return_nan) {
            output[i] = nan();
        } else {
            const double n = static_cast<double>(history.size());
            const double variance = (sum2 - sum * sum / n) / n;
            output[i] = std::sqrt(variance < 0.0 && variance > -1e-12 ? 0.0 : variance);
        }
    }
}

template <typename Value>
void rolling_variance_fresh(
    const Value *values,
    std::size_t size,
    std::size_t window,
    bool fillna,
    double *output) {
    double sum = 0.0;
    double sum2 = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        const double value = static_cast<double>(values[i]);
        sum += value;
        sum2 += value * value;
        if (i >= window) {
            const double old = static_cast<double>(values[i - window]);
            sum -= old;
            sum2 -= old * old;
        }

        const std::size_t count = std::min(window, i + 1);
        if (!fillna && count < window) {
            output[i] = nan();
        } else {
            const double n = static_cast<double>(count);
            const double variance = (sum2 - sum * sum / n) / n;
            output[i] = variance < 0.0 && variance > -1e-12 ? 0.0 : variance;
        }
    }
}

template <typename Value>
void rolling_stddev_fresh(
    const Value *values,
    std::size_t size,
    std::size_t window,
    bool fillna,
    int window_size,
    double *output) {
    double sum = 0.0;
    double sum2 = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        const double value = static_cast<double>(values[i]);
        sum += value;
        sum2 += value * value;
        if (i >= window) {
            const double old = static_cast<double>(values[i - window]);
            sum -= old;
            sum2 -= old * old;
        }

        if (!fillna && static_cast<long>(i) < window_size) {
            output[i] = nan();
        } else {
            const double n = static_cast<double>(std::min(window, i + 1));
            const double variance = (sum2 - sum * sum / n) / n;
            output[i] = std::sqrt(variance < 0.0 && variance > -1e-12 ? 0.0 : variance);
        }
    }
}

template <typename Value>
void rebuild_buffer_sum(
    const Value *values,
    std::size_t size,
    std::size_t window,
    RollingBuffer &history,
    double &sum,
    double &sum2) {
    history.reset();
    sum = 0.0;
    sum2 = 0.0;
    const std::size_t start = size > window ? size - window : 0;
    for (std::size_t i = start; i < size; ++i) {
        const double value = static_cast<double>(values[i]);
        history.push(value);
        sum += value;
        sum2 += value * value;
    }
}

template <typename Value0, typename Value1>
void rolling_pair_fresh(
    const Value0 *x,
    const Value1 *y,
    std::size_t size,
    std::size_t window,
    bool fillna,
    bool beta,
    double *output) {
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;
    double sum_xy = 0.0;

    for (std::size_t i = 0; i < size; ++i) {
        const double x_value = static_cast<double>(x[i]);
        const double y_value = static_cast<double>(y[i]);
        sum_x += x_value;
        sum_y += y_value;
        sum_x2 += x_value * x_value;
        sum_y2 += y_value * y_value;
        sum_xy += x_value * y_value;
        if (i >= window) {
            const double old_x = static_cast<double>(x[i - window]);
            const double old_y = static_cast<double>(y[i - window]);
            sum_x -= old_x;
            sum_y -= old_y;
            sum_x2 -= old_x * old_x;
            sum_y2 -= old_y * old_y;
            sum_xy -= old_x * old_y;
        }

        const std::size_t count = std::min(window, i + 1);
        if (!fillna && count < window) {
            output[i] = nan();
        } else {
            const double n = static_cast<double>(count);
            const double covariance = n * sum_xy - sum_x * sum_y;
            if (beta) {
                output[i] = safe_divide(covariance, n * sum_y2 - sum_y * sum_y);
            } else {
                output[i] = safe_divide(covariance, std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)));
            }
        }
    }
}

template <typename Close, typename High, typename Low>
void true_range(
    const Close *close,
    const High *high,
    const Low *low,
    std::size_t size,
    double &previous_close,
    bool &first,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const double close_value = static_cast<double>(close[i]);
        const double high_value = static_cast<double>(high[i]);
        const double low_value = static_cast<double>(low[i]);
        const double high_low = high_value - low_value;
        if (first) {
            output[i] = high_low;
            first = false;
        } else {
            const double high_close = std::abs(high_value - previous_close);
            const double low_close = std::abs(low_value - previous_close);
            output[i] = std::max(high_low, std::max(high_close, low_close));
        }
        previous_close = close_value;
    }
}

template <typename Value>
void momentum(
    const Value *values,
    std::size_t size,
    const RollingBuffer &history,
    int window,
    bool fillna,
    long start_counter,
    double *output) {
    const std::size_t history_size = history.size();
    const long first_history_index = start_counter - static_cast<long>(history_size);

    for (std::size_t i = 0; i < size; ++i) {
        const long current_index = start_counter + static_cast<long>(i);
        const double value = static_cast<double>(values[i]);
        double previous = nan();
        if (current_index >= window) {
            const long previous_index = current_index - window;
            if (previous_index >= first_history_index && previous_index < start_counter) {
                previous = history.at(static_cast<std::size_t>(previous_index - first_history_index));
            } else {
                previous = static_cast<double>(values[static_cast<std::size_t>(previous_index - start_counter)]);
            }
        }

        const bool return_nan = !fillna && current_index < window;
        output[i] = return_nan || std::isnan(previous) ? nan() : value - previous;
    }
}

template <typename Value>
void chande_momentum_oscillator(
    const Value *values,
    std::size_t size,
    RollingBuffer &gains,
    RollingBuffer &losses,
    double &gain_sum,
    double &loss_sum,
    double &previous,
    bool &first,
    bool fillna,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        double gain = 0.0;
        double loss = 0.0;
        const double close_value = static_cast<double>(values[i]);
        if (!first) {
            const double change = close_value - previous;
            if (change > 0.0) {
                gain = change;
            } else {
                loss = -change;
            }
        }

        rolling_sum_push(gains, gain_sum, gain);
        rolling_sum_push(losses, loss_sum, loss);
        previous = close_value;
        first = false;
        output[i] = (!fillna && !gains.full()) ? nan() : safe_divide(100.0 * (gain_sum - loss_sum), gain_sum + loss_sum);
    }
}

template <typename Value>
void chande_momentum_oscillator_fresh(
    const Value *values,
    std::size_t size,
    std::size_t window,
    bool fillna,
    double *output) {
    double gain_sum = 0.0;
    double loss_sum = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        double gain = 0.0;
        double loss = 0.0;
        if (i > 0) {
            const double change = static_cast<double>(values[i]) - static_cast<double>(values[i - 1]);
            if (change > 0.0) {
                gain = change;
            } else {
                loss = -change;
            }
        }
        gain_sum += gain;
        loss_sum += loss;
        if (i >= window) {
            const double old_change = static_cast<double>(values[i - window + 1]) - static_cast<double>(values[i - window]);
            if (old_change > 0.0) {
                gain_sum -= old_change;
            } else {
                loss_sum += old_change;
            }
        }

        const std::size_t count = std::min(window, i + 1);
        output[i] = (!fillna && count < window) ? nan() : safe_divide(100.0 * (gain_sum - loss_sum), gain_sum + loss_sum);
    }
}

template <typename Close, typename High, typename Low, typename Volume>
void money_flow_index(
    const Close *close,
    const High *high,
    const Low *low,
    const Volume *volume,
    std::size_t size,
    RollingBuffer &positive_history,
    RollingBuffer &negative_history,
    double &positive_sum,
    double &negative_sum,
    double &previous_typical,
    bool &first,
    bool fillna,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const double typical = (
            static_cast<double>(high[i]) +
            static_cast<double>(low[i]) +
            static_cast<double>(close[i])) / 3.0;
        const double money_flow = typical * static_cast<double>(volume[i]);
        double positive = 0.0;
        double negative = 0.0;

        if (!first) {
            if (typical > previous_typical) {
                positive = money_flow;
            } else if (typical < previous_typical) {
                negative = money_flow;
            }
        }

        rolling_sum_push(positive_history, positive_sum, positive);
        rolling_sum_push(negative_history, negative_sum, negative);
        previous_typical = typical;
        first = false;

        if (!fillna && !positive_history.full()) {
            output[i] = nan();
        } else if (negative_sum == 0.0) {
            output[i] = positive_sum == 0.0 ? 50.0 : 100.0;
        } else {
            output[i] = 100.0 - 100.0 / (1.0 + positive_sum / negative_sum);
        }
    }
}

template <typename Close, typename High, typename Low, typename Volume>
void money_flow_index_fresh(
    const Close *close,
    const High *high,
    const Low *low,
    const Volume *volume,
    std::size_t size,
    std::size_t window,
    bool fillna,
    double *output) {
    double positive_sum = 0.0;
    double negative_sum = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        const double typical = (static_cast<double>(high[i]) + static_cast<double>(low[i]) + static_cast<double>(close[i])) / 3.0;
        double positive = 0.0;
        double negative = 0.0;
        if (i > 0) {
            const double previous_typical = (static_cast<double>(high[i - 1]) + static_cast<double>(low[i - 1]) + static_cast<double>(close[i - 1])) / 3.0;
            const double money_flow = typical * static_cast<double>(volume[i]);
            if (typical > previous_typical) {
                positive = money_flow;
            } else if (typical < previous_typical) {
                negative = money_flow;
            }
        }

        positive_sum += positive;
        negative_sum += negative;
        if (i >= window) {
            const std::size_t old_index = i - window;
            const double old_typical = (
                static_cast<double>(high[old_index]) +
                static_cast<double>(low[old_index]) +
                static_cast<double>(close[old_index])) / 3.0;
            const double old_previous_typical = old_index == 0 ? old_typical : (
                static_cast<double>(high[old_index - 1]) +
                static_cast<double>(low[old_index - 1]) +
                static_cast<double>(close[old_index - 1])) / 3.0;
            const double old_money_flow = old_typical * static_cast<double>(volume[old_index]);
            if (old_typical > old_previous_typical) {
                positive_sum -= old_money_flow;
            } else if (old_typical < old_previous_typical) {
                negative_sum -= old_money_flow;
            }
        }

        const std::size_t count = std::min(window, i + 1);
        if (!fillna && count < window) {
            output[i] = nan();
        } else if (negative_sum == 0.0) {
            output[i] = positive_sum == 0.0 ? 50.0 : 100.0;
        } else {
            output[i] = 100.0 - 100.0 / (1.0 + positive_sum / negative_sum);
        }
    }
}

template <typename Value>
void rolling_extreme_value(
    const Value *values,
    std::size_t size,
    RollingExtreme &extreme,
    int window,
    bool fillna,
    long &counter,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        extreme.push(static_cast<double>(values[i]));
        const bool return_nan = !fillna && counter < window;
        ++counter;
        output[i] = return_nan ? nan() : extreme.value();
    }
}

template <typename Value>
void rolling_extreme_index(
    const Value *values,
    std::size_t size,
    RollingExtreme &extreme,
    bool fillna,
    long &counter,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        ++counter;
        extreme.push(static_cast<double>(values[i]));
        if (!fillna && !extreme.full()) {
            output[i] = nan();
        } else {
            const long base = counter - static_cast<long>(extreme.size()) + 1;
            output[i] = static_cast<double>(base + static_cast<long>(extreme.offset()));
        }
    }
}

template <typename Value>
void midpoint(
    const Value *values,
    std::size_t size,
    RollingExtreme &minimum,
    RollingExtreme &maximum,
    bool fillna,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const double value = static_cast<double>(values[i]);
        minimum.push(value);
        maximum.push(value);
        output[i] = (!fillna && !minimum.full()) ? nan() : (minimum.value() + maximum.value()) * 0.5;
    }
}

template <typename High, typename Low>
void midprice(
    const High *high,
    const Low *low,
    std::size_t size,
    RollingExtreme &highs,
    RollingExtreme &lows,
    bool fillna,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        highs.push(static_cast<double>(high[i]));
        lows.push(static_cast<double>(low[i]));
        output[i] = (!fillna && !highs.full()) ? nan() : (highs.value() + lows.value()) * 0.5;
    }
}

template <typename High, typename Low>
void midprice_small_window_scan(
    const High *high,
    const Low *low,
    std::size_t size,
    const double *prior_high,
    const double *prior_low,
    std::size_t prior_size,
    std::size_t window,
    bool fillna,
    double *output) {
    if (prior_size == 0) {
        for (std::size_t i = 0; i < size; ++i) {
            const std::size_t count = std::min(window, i + 1);
            if (!fillna && count < window) {
                output[i] = nan();
                continue;
            }

            const std::size_t start = i + 1 - count;
            double highest = static_cast<double>(high[start]);
            double lowest = static_cast<double>(low[start]);
            for (std::size_t j = start + 1; j <= i; ++j) {
                highest = std::max(highest, static_cast<double>(high[j]));
                lowest = std::min(lowest, static_cast<double>(low[j]));
            }
            output[i] = (highest + lowest) * 0.5;
        }
        return;
    }

    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t current = prior_size + i;
        const std::size_t count = std::min(window, current + 1);
        if (!fillna && count < window) {
            output[i] = nan();
            continue;
        }

        const std::size_t start = current + 1 - count;
        double highest = -std::numeric_limits<double>::infinity();
        double lowest = std::numeric_limits<double>::infinity();
        for (std::size_t j = start; j <= current; ++j) {
            const double high_value = j < prior_size ? prior_high[j] : static_cast<double>(high[j - prior_size]);
            const double low_value = j < prior_size ? prior_low[j] : static_cast<double>(low[j - prior_size]);
            highest = std::max(highest, high_value);
            lowest = std::min(lowest, low_value);
        }
        output[i] = (highest + lowest) * 0.5;
    }
}

template <typename Value>
void rolling_high_small_window_scan(
    const Value *values,
    std::size_t size,
    std::size_t window,
    bool fillna,
    bool high_fillna_requires_extra_sample,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t count = std::min(window, i + 1);
        if (!fillna && (high_fillna_requires_extra_sample ? i < window : count < window)) {
            output[i] = nan();
            continue;
        }

        const std::size_t start = i + 1 - count;
        double highest = static_cast<double>(values[start]);
        for (std::size_t j = start + 1; j <= i; ++j) {
            highest = std::max(highest, static_cast<double>(values[j]));
        }
        output[i] = highest;
    }
}

template <typename Value>
void rolling_low_small_window_scan(
    const Value *values,
    std::size_t size,
    std::size_t window,
    bool fillna,
    bool high_fillna_requires_extra_sample,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t count = std::min(window, i + 1);
        if (!fillna && (high_fillna_requires_extra_sample ? i < window : count < window)) {
            output[i] = nan();
            continue;
        }

        const std::size_t start = i + 1 - count;
        double lowest = static_cast<double>(values[start]);
        for (std::size_t j = start + 1; j <= i; ++j) {
            lowest = std::min(lowest, static_cast<double>(values[j]));
        }
        output[i] = lowest;
    }
}

template <typename Value>
void high_low_small_window_scan(
    const Value *values,
    std::size_t size,
    std::size_t window,
    bool fillna,
    double *min_output,
    double *max_output) {
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t count = std::min(window, i + 1);
        if (!fillna && count < window) {
            min_output[i] = nan();
            max_output[i] = nan();
            continue;
        }

        const std::size_t start = i + 1 - count;
        double lowest = static_cast<double>(values[start]);
        double highest = lowest;
        for (std::size_t j = start + 1; j <= i; ++j) {
            const double value = static_cast<double>(values[j]);
            highest = std::max(highest, value);
            lowest = std::min(lowest, value);
        }
        min_output[i] = lowest;
        max_output[i] = highest;
    }
}

template <typename Value>
void high_low_index_small_window_scan(
    const Value *values,
    std::size_t size,
    std::size_t window,
    bool fillna,
    double *min_index,
    double *max_index) {
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t count = std::min(window, i + 1);
        if (!fillna && count < window) {
            min_index[i] = nan();
            max_index[i] = nan();
            continue;
        }

        const std::size_t start = i + 1 - count;
        double lowest = static_cast<double>(values[start]);
        double highest = lowest;
        std::size_t lowest_index = start;
        std::size_t highest_index = start;
        for (std::size_t j = start + 1; j <= i; ++j) {
            const double value = static_cast<double>(values[j]);
            if (value <= lowest) {
                lowest = value;
                lowest_index = j;
            }
            if (value >= highest) {
                highest = value;
                highest_index = j;
            }
        }
        min_index[i] = static_cast<double>(lowest_index);
        max_index[i] = static_cast<double>(highest_index);
    }
}

template <typename High, typename Low>
void aroon_small_window_scan(
    const High *high,
    const Low *low,
    std::size_t size,
    std::size_t window,
    bool fillna,
    double *down,
    double *up) {
    const std::size_t lookback = window + 1;
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t count = std::min(lookback, i + 1);
        if (!fillna && count < lookback) {
            down[i] = nan();
            up[i] = nan();
            continue;
        }

        const std::size_t start = i + 1 - count;
        double highest = static_cast<double>(high[start]);
        double lowest = static_cast<double>(low[start]);
        std::size_t highest_index = start;
        std::size_t lowest_index = start;
        for (std::size_t j = start + 1; j <= i; ++j) {
            const double high_value = static_cast<double>(high[j]);
            const double low_value = static_cast<double>(low[j]);
            if (high_value >= highest) {
                highest = high_value;
                highest_index = j;
            }
            if (low_value <= lowest) {
                lowest = low_value;
                lowest_index = j;
            }
        }

        const double denom = static_cast<double>(std::max<std::size_t>(count - 1, 1));
        down[i] = 100.0 * static_cast<double>(lowest_index - start) / denom;
        up[i] = 100.0 * static_cast<double>(highest_index - start) / denom;
    }
}

template <typename Close, typename High, typename Low>
void williams_r_small_window_scan(
    const Close *close,
    const High *high,
    const Low *low,
    std::size_t size,
    std::size_t window,
    bool fillna,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t count = std::min(window, i + 1);
        if (!fillna && count < window) {
            output[i] = nan();
            continue;
        }

        const std::size_t start = i + 1 - count;
        double highest = static_cast<double>(high[start]);
        double lowest = static_cast<double>(low[start]);
        for (std::size_t j = start + 1; j <= i; ++j) {
            highest = std::max(highest, static_cast<double>(high[j]));
            lowest = std::min(lowest, static_cast<double>(low[j]));
        }
        output[i] = safe_divide(-100.0 * (highest - static_cast<double>(close[i])), highest - lowest);
    }
}

template <typename Close, typename High, typename Low>
void stochastic_fastk_small_window_scan(
    const Close *close,
    const High *high,
    const Low *low,
    std::size_t size,
    std::size_t window,
    bool fillna,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t count = std::min(window, i + 1);
        if (!fillna && count < window) {
            output[i] = nan();
            continue;
        }

        const std::size_t start = i + 1 - count;
        double highest = static_cast<double>(high[start]);
        double lowest = static_cast<double>(low[start]);
        for (std::size_t j = start + 1; j <= i; ++j) {
            highest = std::max(highest, static_cast<double>(high[j]));
            lowest = std::min(lowest, static_cast<double>(low[j]));
        }
        output[i] = safe_divide(100.0 * (static_cast<double>(close[i]) - lowest), highest - lowest);
    }
}

template <typename Value>
void rebuild_extreme_state(const Value *values, std::size_t size, std::size_t window, RollingExtreme &extreme) {
    extreme.reset();
    const std::size_t start = size > window ? size - window : 0;
    for (std::size_t i = start; i < size; ++i) {
        extreme.push(static_cast<double>(values[i]));
    }
}

template <typename High, typename Low>
void rebuild_pair_extreme_state(
    const High *high,
    const Low *low,
    std::size_t size,
    std::size_t window,
    RollingExtreme &highs,
    RollingExtreme &lows) {
    highs.reset();
    lows.reset();
    const std::size_t start = size > window ? size - window : 0;
    for (std::size_t i = start; i < size; ++i) {
        highs.push(static_cast<double>(high[i]));
        lows.push(static_cast<double>(low[i]));
    }
}

template <typename High, typename Low>
void aroon(
    const High *high,
    const Low *low,
    std::size_t size,
    RollingExtreme &highs,
    RollingExtreme &lows,
    bool fillna,
    double *down,
    double *up) {
    for (std::size_t i = 0; i < size; ++i) {
        highs.push(static_cast<double>(high[i]));
        lows.push(static_cast<double>(low[i]));
        if (!fillna && !highs.full()) {
            down[i] = nan();
            up[i] = nan();
        } else {
            const double denom = static_cast<double>(std::max<std::size_t>(highs.size() - 1, 1));
            const double periods_since_high = denom - static_cast<double>(highs.offset());
            const double periods_since_low = denom - static_cast<double>(lows.offset());
            down[i] = 100.0 * (denom - periods_since_low) / denom;
            up[i] = 100.0 * (denom - periods_since_high) / denom;
        }
    }
}

template <typename High, typename Low>
void aroon_oscillator(
    const High *high,
    const Low *low,
    std::size_t size,
    RollingExtreme &highs,
    RollingExtreme &lows,
    bool fillna,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        highs.push(static_cast<double>(high[i]));
        lows.push(static_cast<double>(low[i]));
        if (!fillna && !highs.full()) {
            output[i] = nan();
        } else {
            const double denom = static_cast<double>(std::max<std::size_t>(highs.size() - 1, 1));
            const double periods_since_high = denom - static_cast<double>(highs.offset());
            const double periods_since_low = denom - static_cast<double>(lows.offset());
            const double down = 100.0 * (denom - periods_since_low) / denom;
            const double up = 100.0 * (denom - periods_since_high) / denom;
            output[i] = up - down;
        }
    }
}

template <typename Close, typename High, typename Low>
void williams_r(
    const Close *close,
    const High *high,
    const Low *low,
    std::size_t size,
    RollingExtreme &highs,
    RollingExtreme &lows,
    bool fillna,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        highs.push(static_cast<double>(high[i]));
        lows.push(static_cast<double>(low[i]));
        if (!fillna && !highs.full()) {
            output[i] = nan();
        } else {
            const double highest = highs.value();
            const double lowest = lows.value();
            output[i] = safe_divide(-100.0 * (highest - static_cast<double>(close[i])), highest - lowest);
        }
    }
}

template <typename Value>
void high_low(
    const Value *values,
    std::size_t size,
    RollingExtreme &minimum,
    RollingExtreme &maximum,
    bool fillna,
    double *min_output,
    double *max_output) {
    for (std::size_t i = 0; i < size; ++i) {
        const double value = static_cast<double>(values[i]);
        minimum.push(value);
        maximum.push(value);
        if (!fillna && !minimum.full()) {
            min_output[i] = nan();
            max_output[i] = nan();
        } else {
            min_output[i] = minimum.value();
            max_output[i] = maximum.value();
        }
    }
}

template <typename Value>
void high_low_index(
    const Value *values,
    std::size_t size,
    RollingExtreme &minimum,
    RollingExtreme &maximum,
    bool fillna,
    long &counter,
    double *min_index,
    double *max_index) {
    for (std::size_t i = 0; i < size; ++i) {
        ++counter;
        const double value = static_cast<double>(values[i]);
        minimum.push(value);
        maximum.push(value);
        if (!fillna && !minimum.full()) {
            min_index[i] = nan();
            max_index[i] = nan();
        } else {
            const long base = counter - static_cast<long>(minimum.size()) + 1;
            min_index[i] = static_cast<double>(base + static_cast<long>(minimum.offset()));
            max_index[i] = static_cast<double>(base + static_cast<long>(maximum.offset()));
        }
    }
}

template <typename Close, typename High, typename Low>
void ultimate_oscillator(
    const Close *close,
    const High *high,
    const Low *low,
    std::size_t size,
    RollingBuffer &bp_short,
    RollingBuffer &tr_short,
    RollingBuffer &bp_medium,
    RollingBuffer &tr_medium,
    RollingBuffer &bp_long,
    RollingBuffer &tr_long,
    double &bp_short_sum,
    double &tr_short_sum,
    double &bp_medium_sum,
    double &tr_medium_sum,
    double &bp_long_sum,
    double &tr_long_sum,
    double &previous_close,
    bool &first,
    bool fillna,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const double close_value = static_cast<double>(close[i]);
        const double high_value = static_cast<double>(high[i]);
        const double low_value = static_cast<double>(low[i]);
        const double prev = first ? close_value : previous_close;
        const double min_low_prev = std::min(low_value, prev);
        const double buying_pressure = close_value - min_low_prev;
        const double range = std::max(high_value, prev) - min_low_prev;

        rolling_sum_push(bp_short, bp_short_sum, buying_pressure);
        rolling_sum_push(tr_short, tr_short_sum, range);
        rolling_sum_push(bp_medium, bp_medium_sum, buying_pressure);
        rolling_sum_push(tr_medium, tr_medium_sum, range);
        rolling_sum_push(bp_long, bp_long_sum, buying_pressure);
        rolling_sum_push(tr_long, tr_long_sum, range);

        previous_close = close_value;
        first = false;

        if (!fillna && !bp_long.full()) {
            output[i] = nan();
        } else {
            const double avg_short = safe_divide(bp_short_sum, tr_short_sum);
            const double avg_medium = safe_divide(bp_medium_sum, tr_medium_sum);
            const double avg_long = safe_divide(bp_long_sum, tr_long_sum);
            output[i] = 100.0 * (4.0 * avg_short + 2.0 * avg_medium + avg_long) / 7.0;
        }
    }
}

}  // namespace batch_kernels

class WilderSmoothing {
public:
    explicit WilderSmoothing(int window)
        : window_(window),
          value_(0.0),
          count_(0) {}

    double update(double value) {
        value_ = (value_ * (window_ - 1.0) + value) / window_;
        ++count_;
        return value_;
    }

    int window() const {
        return window_;
    }

    double value() const {
        return value_;
    }

    long count() const {
        return count_;
    }

    void set_state(double value, long count) {
        value_ = value;
        count_ = count;
    }

private:
    int window_;
    double value_;
    long count_;
};

class Delay {
public:
    explicit Delay(int window = 1, bool fillna = true)
        : index_(0),
          max_(window),
          fillna_(fillna),
          buffer_(static_cast<std::size_t>(window), 0.0),
          first_(true) {}

    double update(double value) {
        if (first_) {
            first_ = false;
            std::fill(buffer_.begin(), buffer_.end(), fillna_ ? 0.0 : nan());
        }

        const double retval = buffer_[static_cast<std::size_t>(index_)];
        buffer_[static_cast<std::size_t>(index_)] = value;

        ++index_;
        if (index_ == max_) {
            index_ = 0;
        }

        return retval;
    }

    double peek() const {
        return buffer_[static_cast<std::size_t>(index_)];
    }

private:
    int index_;
    int max_;
    bool fillna_;
    std::vector<double> buffer_;
    bool first_;
};

namespace batch_kernels {

template <typename Value>
void rate_of_change(
    const Value *values,
    std::size_t size,
    Delay &history,
    int window,
    bool fillna,
    long &counter,
    double scale,
    bool subtract_previous,
    double *output) {
    for (std::size_t i = 0; i < size; ++i) {
        const double value = static_cast<double>(values[i]);
        const double previous = history.update(value);
        const bool return_nan = !fillna && counter < window;
        ++counter;
        output[i] = return_nan ? nan() : safe_divide(subtract_previous ? value - previous : value, previous) * scale;
    }
}

}  // namespace batch_kernels

class ATR {
public:
    explicit ATR(double window = 14.0, bool fillna = true)
        : window_(std::max(static_cast<int>(window), 1)),
          fillna_(fillna),
          previous_close_(0.0),
          first_(true),
          count_(0),
          tr_sum_(0.0),
          atr_(0.0) {}

    double update(double close, double high, double low) {
        const double tr = first_ ? high - low : true_range(close, high, low, previous_close_);
        previous_close_ = close;
        first_ = false;
        ++count_;

        if (count_ <= window_) {
            tr_sum_ += tr;
            atr_ = tr_sum_ / static_cast<double>(count_);
            return (!fillna_ && count_ < window_) ? nan() : atr_;
        }

        atr_ = (atr_ * (window_ - 1.0) + tr) / window_;
        return atr_;
    }

    template <typename Array0, typename Array1, typename Array2>
    void batch_to(const Array0 &close, const Array1 &high, const Array2 &low, double *output) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        const auto *close_values = close.data();
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        double previous_close = previous_close_;
        bool first = first_;
        int count = count_;
        double tr_sum = tr_sum_;
        double atr = atr_;

        for (std::size_t i = 0; i < size; ++i) {
            const double close_value = static_cast<double>(close_values[i]);
            const double high_value = static_cast<double>(high_values[i]);
            const double low_value = static_cast<double>(low_values[i]);
            const double tr = first ? high_value - low_value : true_range(close_value, high_value, low_value, previous_close);
            previous_close = close_value;
            first = false;
            ++count;

            if (count <= window_) {
                tr_sum += tr;
                atr = tr_sum / static_cast<double>(count);
                output[i] = (!fillna_ && count < window_) ? nan() : atr;
            } else {
                atr = (atr * (window_ - 1.0) + tr) / window_;
                output[i] = atr;
            }
        }

        previous_close_ = previous_close;
        first_ = first;
        count_ = count;
        tr_sum_ = tr_sum;
        atr_ = atr;
    }

    template <typename Array0, typename Array1, typename Array2>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        batch_to(close, high, low, output.data());
        return make_array(std::move(output));
    }

private:
    int window_;
    bool fillna_;
    double previous_close_;
    bool first_;
    int count_;
    double tr_sum_;
    double atr_;
};

class ATRP {
public:
    explicit ATRP(double window = 14.0, bool fillna = true)
        : atr_(window, fillna) {}

    double update(double close, double high, double low) {
        return atr_.update(close, high, low) / close;
    }

    template <typename Array0, typename Array1, typename Array2>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        atr_.batch_to(close, high, low, output.data());
        const auto *close_values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] /= static_cast<double>(close_values[i]);
        }
        return make_array(std::move(output));
    }

private:
    ATR atr_;
};

class SuperTrend {
public:
    SuperTrend(int window = 10, double multiplier = 3.0, bool fillna = true)
        : atr_(window, fillna),
          multiplier_(multiplier),
          previous_close_(0.0),
          upper_(nan()),
          lower_(nan()),
          first_(true),
          fillna_(fillna),
          last_{nan(), nan(), nan(), nan()} {}

    SuperTrendResult update(double close, double high, double low) {
        update_core(close, high, low);
        return last_;
    }

    void advance(double close, double high, double low) {
        update_core(close, high, low);
    }

    inline const SuperTrendResult &last() const {
        return last_;
    }

    template <typename Array0, typename Array1, typename Array2>
    SuperTrendBatchResult batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> value(size);
        std::vector<double> direction(size);
        std::vector<double> upper(size);
        std::vector<double> lower(size);
        const auto *close_values = close.data();
        const auto *high_values = high.data();
        const auto *low_values = low.data();

        for (std::size_t i = 0; i < size; ++i) {
            const SuperTrendResult out = update(
                static_cast<double>(close_values[i]),
                static_cast<double>(high_values[i]),
                static_cast<double>(low_values[i]));
            value[i] = out.value;
            direction[i] = out.direction;
            upper[i] = out.upper;
            lower[i] = out.lower;
        }

        return {
            make_array(std::move(value)),
            make_array(std::move(direction)),
            make_array(std::move(upper)),
            make_array(std::move(lower)),
        };
    }

private:
    inline void update_core(double close, double high, double low) {
        const double atr = atr_.update(close, high, low);
        const double midpoint = (high + low) * 0.5;
        const double basic_upper = midpoint + multiplier_ * atr;
        const double basic_lower = midpoint - multiplier_ * atr;

        if (!fillna_ && std::isnan(atr)) {
            previous_close_ = close;
            last_ = {nan(), nan(), nan(), nan()};
            return;
        }

        if (first_) {
            upper_ = basic_upper;
            lower_ = basic_lower;
            const bool uptrend = close >= midpoint;
            last_ = {uptrend ? lower_ : upper_, uptrend ? 1.0 : -1.0, upper_, lower_};
            previous_close_ = close;
            first_ = false;
            return;
        }

        const double previous_upper = upper_;
        const double previous_lower = lower_;
        const double previous_value = last_.value;
        upper_ = (basic_upper < previous_upper || previous_close_ > previous_upper) ? basic_upper : previous_upper;
        lower_ = (basic_lower > previous_lower || previous_close_ < previous_lower) ? basic_lower : previous_lower;

        bool uptrend;
        if (previous_value == previous_upper) {
            uptrend = close > upper_;
        } else {
            uptrend = close >= lower_;
        }

        last_ = {uptrend ? lower_ : upper_, uptrend ? 1.0 : -1.0, upper_, lower_};
        previous_close_ = close;
    }

    ATR atr_;
    double multiplier_;
    double previous_close_;
    double upper_;
    double lower_;
    bool first_;
    bool fillna_;
    SuperTrendResult last_;
};

class EMA {
public:
    explicit EMA(double window = 1.0, bool fillna = false)
        : first_pass_(true),
          index_(0),
          window_(std::max(static_cast<int>(window), 1)),
          fillna_(fillna),
          last_value_(0.0),
          weighted_multiplier_(2.0 / (1.0 + window)),
          inverted_multiplier_(1.0 - weighted_multiplier_) {}

    double update(double value) {
        if (first_pass_ && index_ == 0) {
            last_value_ = value;
        }

        last_value_ = weighted_multiplier_ * value + last_value_ * inverted_multiplier_;
        ++index_;

        if (index_ == window_) {
            first_pass_ = false;
        }

        if (first_pass_ && !fillna_) {
            return nan();
        }

        return last_value_;
    }

    inline double update_always_filled(double value) {
        if (first_pass_ && index_ == 0) {
            last_value_ = value;
        }

        last_value_ = weighted_multiplier_ * value + last_value_ * inverted_multiplier_;
        ++index_;

        if (index_ == window_) {
            first_pass_ = false;
        }

        return last_value_;
    }

    nb::object batch(const InputArray &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        const double *values = input.data();

        double last = last_value_;
        long index = index_;
        bool first_pass = first_pass_;
        std::size_t i = 0;

        if (first_pass) {
            for (; i < size && first_pass; ++i) {
                const double value = values[i];
                if (index == 0) {
                    last = value;
                }

                last = weighted_multiplier_ * value + last * inverted_multiplier_;
                ++index;

                if (index == window_) {
                    first_pass = false;
                }

                output[i] = (!fillna_ && first_pass) ? nan() : last;
            }
        }

        for (; i < size; ++i) {
            last = weighted_multiplier_ * values[i] + last * inverted_multiplier_;
            ++index;
            output[i] = last;
        }

        last_value_ = last;
        index_ = index;
        first_pass_ = first_pass;
        return make_array(std::move(output));
    }

private:
    bool first_pass_;
    long index_;
    int window_;
    bool fillna_;
    double last_value_;
    double weighted_multiplier_;
    double inverted_multiplier_;
};

namespace batch_kernels {

template <typename Value>
void t3_moving_average(
    const Value *values,
    std::size_t size,
    EMA &e1_indicator,
    EMA &e2_indicator,
    EMA &e3_indicator,
    EMA &e4_indicator,
    EMA &e5_indicator,
    EMA &e6_indicator,
    double vfactor,
    int window,
    bool fillna,
    long &counter,
    double *output) {
    const double v2 = vfactor * vfactor;
    const double v3 = v2 * vfactor;
    const double c1 = -v3;
    const double c2 = 3.0 * v2 + 3.0 * v3;
    const double c3 = -6.0 * v2 - 3.0 * vfactor - 3.0 * v3;
    const double c4 = 1.0 + 3.0 * vfactor + v3 + 3.0 * v2;

    for (std::size_t i = 0; i < size; ++i) {
        const double e1 = e1_indicator.update_always_filled(static_cast<double>(values[i]));
        const double e2 = e2_indicator.update_always_filled(e1);
        const double e3 = e3_indicator.update_always_filled(e2);
        const double e4 = e4_indicator.update_always_filled(e3);
        const double e5 = e5_indicator.update_always_filled(e4);
        const double e6 = e6_indicator.update_always_filled(e5);
        const bool return_nan = !fillna && counter < window;
        ++counter;
        output[i] = return_nan ? nan() : c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
    }
}

}  // namespace batch_kernels

class SMA {
public:
    explicit SMA(int window = 1, bool fillna = false)
        : first_pass_(true),
          history_(static_cast<std::size_t>(window), 0.0),
          index_(0),
          window_(window),
          fillna_(fillna),
          tally_(0.0),
          stored_mean_(nan()),
          mean_(nan()) {}

    int length() const {
        return window_;
    }

    double mean() const {
        return stored_mean_;
    }

    double update(double value) {
        tally_ -= history_[static_cast<std::size_t>(index_)];
        tally_ += value;
        history_[static_cast<std::size_t>(index_)] = value;

        ++index_;

        if (index_ == window_) {
            index_ = 0;
            first_pass_ = false;
        }

        if (first_pass_) {
            if (!fillna_) {
                mean_ = nan();
                stored_mean_ = mean_;
                return mean_;
            }
            mean_ = tally_ / index_;
            stored_mean_ = mean_;
            return mean_;
        }

        mean_ = tally_ / window_;
        stored_mean_ = mean_;
        return mean_;
    }

    nb::object batch(const InputArray &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        const double *values = input.data();
        bool first_pass = first_pass_;
        int index = index_;
        double tally = tally_;
        double mean = mean_;

        for (std::size_t i = 0; i < size; ++i) {
            tally -= history_[static_cast<std::size_t>(index)];
            tally += values[i];
            history_[static_cast<std::size_t>(index)] = values[i];

            ++index;

            if (index == window_) {
                index = 0;
                first_pass = false;
            }

            if (first_pass) {
                mean = fillna_ ? tally / index : nan();
            } else {
                mean = tally / window_;
            }
            output[i] = mean;
        }

        first_pass_ = first_pass;
        index_ = index;
        tally_ = tally;
        mean_ = mean;
        stored_mean_ = mean;
        return make_array(std::move(output));
    }

private:
    bool first_pass_;
    std::vector<double> history_;
    int index_;
    int window_;
    bool fillna_;
    double tally_;
    double stored_mean_;
    double mean_;
};

namespace batch_kernels {

template <typename Close, typename High, typename Low>
void fast_stochastic(
    const Close *close,
    const High *high,
    const Low *low,
    std::size_t size,
    RollingExtreme &highs,
    RollingExtreme &lows,
    SMA &fastd_indicator,
    bool fillna,
    double *fastk,
    double *fastd) {
    for (std::size_t i = 0; i < size; ++i) {
        highs.push(static_cast<double>(high[i]));
        lows.push(static_cast<double>(low[i]));
        const double k = (!fillna && !highs.full()) ? nan() :
            safe_divide(100.0 * (static_cast<double>(close[i]) - lows.value()), highs.value() - lows.value());
        fastk[i] = k;
        fastd[i] = fastd_indicator.update(k);
    }
}

template <typename Close, typename High, typename Low>
void stochastic(
    const Close *close,
    const High *high,
    const Low *low,
    std::size_t size,
    RollingExtreme &highs,
    RollingExtreme &lows,
    SMA &slowk_indicator,
    SMA &slowd_indicator,
    bool fillna,
    double *slowk,
    double *slowd) {
    for (std::size_t i = 0; i < size; ++i) {
        highs.push(static_cast<double>(high[i]));
        lows.push(static_cast<double>(low[i]));
        const double fastk = (!fillna && !highs.full()) ? nan() :
            safe_divide(100.0 * (static_cast<double>(close[i]) - lows.value()), highs.value() - lows.value());
        const double k = slowk_indicator.update(fastk);
        slowk[i] = k;
        slowd[i] = slowd_indicator.update(k);
    }
}

}  // namespace batch_kernels

class AwesomeOscillator {
public:
    AwesomeOscillator(int window_1 = 34, int window_2 = 5, bool fillna = true)
        : oscillator_1_(window_1, true),
          oscillator_2_(window_2, true),
          counter_(0),
          window_(fillna ? 0 : std::max(window_1, window_2)) {}

    double update(double high, double low) {
        ++counter_;

        if (high < low) {
            std::swap(high, low);
        }

        const double median = (high + low) * 0.5;
        const double retval = oscillator_2_.update(median) - oscillator_1_.update(median);

        if (counter_ <= window_) {
            return nan();
        }

        return retval;
    }

    nb::object batch(const InputArray &high, const InputArray &low) {
        const std::size_t size = high.shape(0);
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        const double *high_values = high.data();
        const double *low_values = low.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(high_values[i], low_values[i]);
        }
        return make_array(std::move(output));
    }

private:
    SMA oscillator_1_;
    SMA oscillator_2_;
    long counter_;
    int window_;
};

class EWMA {
public:
    EWMA()
        : alpha_(1.0),
          last_(0.0),
          first_(true) {}

    EWMA(nb::object alpha, nb::object span, nb::object com, bool fillna = false)
        : alpha_(0.0),
          last_(0.0),
          first_(true) {
        static_cast<void>(fillna);

        const int supplied = (alpha.is_none() ? 0 : 1) +
                             (span.is_none() ? 0 : 1) +
                             (com.is_none() ? 0 : 1);
        if (supplied != 1) {
            throw nb::value_error("You must define one of alpha, span or com");
        }

        if (!alpha.is_none()) {
            alpha_ = nb::cast<double>(alpha);
        } else if (!span.is_none()) {
            alpha_ = 2.0 / (nb::cast<double>(span) + 1.0);
        } else {
            alpha_ = 1.0 / (nb::cast<double>(com) + 1.0);
        }

        if (!(0.0 < alpha_ && alpha_ <= 1.0)) {
            throw nb::value_error("EWMA's alpha parameter must be in the range 0<alpha<=1");
        }
    }

    double update(double value) {
        if (first_) {
            last_ = value;
            first_ = false;
        } else {
            last_ = alpha_ * value + (1.0 - alpha_) * last_;
        }
        return last_;
    }

private:
    double alpha_;
    double last_;
    bool first_;
};

class Summation {
public:
    explicit Summation(int window = 1, bool fillna = true)
        : first_pass_(true),
          history_(static_cast<std::size_t>(window), 0.0),
          index_(0),
          window_(window),
          fillna_(fillna),
          tally_(0.0) {}

    double update(double value) {
        tally_ -= history_[static_cast<std::size_t>(index_)];
        tally_ += value;
        history_[static_cast<std::size_t>(index_)] = value;

        ++index_;

        if (index_ == window_) {
            index_ = 0;
            first_pass_ = false;
        }

        if (first_pass_) {
            if (!fillna_) {
                return nan();
            }
            return tally_;
        }

        return tally_;
    }

    nb::object batch(const InputArray &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        const double *values = input.data();
        bool first_pass = first_pass_;
        int index = index_;
        double tally = tally_;

        for (std::size_t i = 0; i < size; ++i) {
            tally -= history_[static_cast<std::size_t>(index)];
            tally += values[i];
            history_[static_cast<std::size_t>(index)] = values[i];

            ++index;

            if (index == window_) {
                index = 0;
                first_pass = false;
            }

            output[i] = (first_pass && !fillna_) ? nan() : tally;
        }

        first_pass_ = first_pass;
        index_ = index;
        tally_ = tally;
        return make_array(std::move(output));
    }

private:
    bool first_pass_;
    std::vector<double> history_;
    int index_;
    int window_;
    bool fillna_;
    double tally_;
};

class Kama {
public:
    Kama(int window = 10, int fast_ema = 2, int slow_ema = 30, bool fillna = true)
        : previous_close_(0.0),
          has_previous_close_(false),
          window_history_(static_cast<std::size_t>(std::max(window, 1)), 0.0),
          window_index_(0),
          den_history_(static_cast<std::size_t>(std::max(window, 1)), 0.0),
          den_index_(0),
          den_tally_(0.0),
          pow1_(fast_ema),
          pow2_(slow_ema),
          first_(true),
          previous_kama_(0.0) {
        static_cast<void>(fillna);
    }

    double update(double close) {
        return update_value(close);
    }

    nb::object batch(const InputArray &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        const double *values = input.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update_value(values[i]);
        }
        return make_array(std::move(output));
    }

private:
    double update_value(double close) {
        const double delayed_close = has_previous_close_ ? previous_close_ : 0.0;
        const double vol = std::abs(close - delayed_close);
        previous_close_ = close;
        has_previous_close_ = true;

        const std::size_t window_index = static_cast<std::size_t>(window_index_);
        const double window_ago = window_history_[window_index];
        window_history_[window_index] = close;
        ++window_index_;
        if (window_index_ == static_cast<int>(window_history_.size())) {
            window_index_ = 0;
        }

        const std::size_t den_index = static_cast<std::size_t>(den_index_);
        den_tally_ -= den_history_[den_index];
        den_tally_ += vol;
        den_history_[den_index] = vol;
        ++den_index_;
        if (den_index_ == static_cast<int>(den_history_.size())) {
            den_index_ = 0;
        }

        const double er_num = std::abs(close - window_ago);
        const double efficiency_ratio = den_tally_ == 0.0 ? 0.0 : er_num / den_tally_;
        const double smoothing_constant =
            std::pow(
                efficiency_ratio * (2.0 / (pow1_ + 1.0) - 2.0 / (pow2_ + 1.0)) +
                    2.0 / (pow2_ + 1.0),
                2.0);

        if (std::isnan(smoothing_constant)) {
            return nan();
        }

        if (first_) {
            previous_kama_ = close;
            first_ = false;
            return close;
        }

        previous_kama_ = previous_kama_ + smoothing_constant * (close - previous_kama_);
        return previous_kama_;
    }

    double previous_close_;
    bool has_previous_close_;
    std::vector<double> window_history_;
    int window_index_;
    std::vector<double> den_history_;
    int den_index_;
    double den_tally_;
    double pow1_;
    double pow2_;
    bool first_;
    double previous_kama_;
};

class VariableIndexDynamicAverage {
public:
    VariableIndexDynamicAverage(int cmo_window = 9, int ema_window = 12, bool fillna = true)
        : gains_(cmo_window),
          losses_(cmo_window),
          smoothing_factor_(2.0 / (static_cast<double>(std::max(ema_window, 1)) + 1.0)),
          gain_sum_(0.0),
          loss_sum_(0.0),
          previous_(0.0),
          value_(0.0),
          first_(true),
          fillna_(fillna) {}

    double update(double close) {
        double gain = 0.0;
        double loss = 0.0;
        if (!first_) {
            const double change = close - previous_;
            if (change > 0.0) {
                gain = change;
            } else {
                loss = -change;
            }
        }

        rolling_sum_push(gains_, gain_sum_, gain);
        rolling_sum_push(losses_, loss_sum_, loss);

        if (first_) {
            value_ = close;
            previous_ = close;
            first_ = false;
            return (!fillna_ && !gains_.full()) ? nan() : value_;
        }

        const double cmo = std::abs(safe_divide(gain_sum_ - loss_sum_, gain_sum_ + loss_sum_));
        const double alpha = smoothing_factor_ * cmo;
        value_ = close * alpha + value_ * (1.0 - alpha);
        previous_ = close;

        if (!fillna_ && !gains_.full()) {
            return nan();
        }
        return value_;
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(values[i]));
        }
        return make_array(std::move(output));
    }

private:
    RollingBuffer gains_;
    RollingBuffer losses_;
    double smoothing_factor_;
    double gain_sum_;
    double loss_sum_;
    double previous_;
    double value_;
    bool first_;
    bool fillna_;
};

struct KalmanMovingAverageTuning {
    double initial_price;
    double initial_velocity;
    double dt;
    double position_variance;
    double velocity_variance;
    double process_position_variance;
    double process_velocity_variance;
    double measurement_variance;
};

struct KalmanLocalLinearTrendTuning {
    double initial_level;
    double initial_trend;
    double dt;
    double level_variance;
    double trend_variance;
    double process_level_variance;
    double process_trend_variance;
    double observation_variance;
};

struct KalmanVelocityOscillatorTuning {
    double initial_price;
    double initial_velocity;
    double dt;
    double position_variance;
    double velocity_variance;
    double process_position_variance;
    double process_velocity_variance;
    double measurement_variance;
};

struct KalmanInnovationZScoreTuning {
    double initial_price;
    double initial_velocity;
    double dt;
    double position_variance;
    double velocity_variance;
    double process_position_variance;
    double process_velocity_variance;
    double measurement_variance;
};

struct KalmanPredictionBandsTuning {
    double initial_price;
    double initial_velocity;
    double dt;
    double position_variance;
    double velocity_variance;
    double process_position_variance;
    double process_velocity_variance;
    double measurement_variance;
};

struct KalmanTrendSignalTuning {
    double initial_price;
    double initial_velocity;
    double dt;
    double position_variance;
    double velocity_variance;
    double process_position_variance;
    double process_velocity_variance;
    double measurement_variance;
};

class KalmanMovingAverage {
public:
    KalmanMovingAverage(double initial_price = nan(),
                        double initial_velocity = 0.0,
                        double dt = 1.0,
                        double position_variance = 1.0,
                        double velocity_variance = 1.0,
                        double process_position_variance = 1.0e-4,
                        double process_velocity_variance = 1.0e-3,
                        double measurement_variance = 0.25,
                        bool fillna = true)
        : initial_price_(initial_price),
          initial_velocity_(initial_velocity),
          dt_(dt > 0.0 ? dt : 1.0),
          position_variance_(variance_floor(position_variance)),
          velocity_variance_(variance_floor(velocity_variance)),
          process_position_variance_(variance_floor(process_position_variance)),
          process_velocity_variance_(variance_floor(process_velocity_variance)),
          measurement_variance_(variance_floor(measurement_variance)),
          initialized_(false),
          count_(0),
          last_(nan()),
          fillna_(fillna) {}

    KalmanMovingAverage(const KalmanMovingAverageTuning &tuning, bool fillna = true)
        : KalmanMovingAverage(
              tuning.initial_price,
              tuning.initial_velocity,
              tuning.dt,
              tuning.position_variance,
              tuning.velocity_variance,
              tuning.process_position_variance,
              tuning.process_velocity_variance,
              tuning.measurement_variance,
              fillna) {}

    double update(double close) {
        if (!initialized_) {
            initialize(std::isfinite(initial_price_) ? initial_price_ : close);
            if (!std::isfinite(initial_price_)) {
                last_ = close;
                ++count_;
                return (!fillna_ && count_ < 2) ? nan() : last_;
            }
        }

        (void) filter_.update(kalman::Vec<1>{close});
        last_ = filter_.x[0];
        ++count_;
        return (!fillna_ && count_ < 2) ? nan() : last_;
    }

    void advance(double close) {
        (void) update(close);
    }

    double last() const {
        return last_;
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(values[i]));
        }
        return make_array(std::move(output));
    }

    template <typename Array>
    static KalmanMovingAverageTuning tune_array(const Array &close, double dt = 1.0, double min_variance = 1.0e-12) {
        const std::size_t size = close.shape(0);
        const auto *values = close.data();
        std::vector<double> prices;
        prices.reserve(size);
        for (std::size_t i = 0; i < size; ++i) {
            const double value = static_cast<double>(values[i]);
            if (std::isfinite(value)) {
                prices.push_back(value);
            }
        }
        return tune_vector(prices, dt, min_variance);
    }

    static KalmanMovingAverageTuning tune_records(nb::iterable records, double dt = 1.0, double min_variance = 1.0e-12) {
        std::vector<double> prices = make_record_output(records);
        for (nb::handle record : records) {
            const double value = scalar_or_record_value(record, "close", 0);
            if (std::isfinite(value)) {
                prices.push_back(value);
            }
        }
        return tune_vector(prices, dt, min_variance);
    }

private:
    static double variance_floor(double value, double minimum = kalman::kDefaultVarianceFloor) {
        if (!std::isfinite(value) || value < minimum) {
            return minimum;
        }
        return value;
    }

    static double median(std::vector<double> values) {
        if (values.empty()) {
            return 0.0;
        }
        std::sort(values.begin(), values.end());
        const std::size_t middle = values.size() / 2;
        if ((values.size() % 2) != 0) {
            return values[middle];
        }
        return 0.5 * (values[middle - 1] + values[middle]);
    }

    static double sample_variance(const std::vector<double> &values) {
        if (values.size() < 2) {
            return 0.0;
        }
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        const double mean = sum / static_cast<double>(values.size());
        double squared = 0.0;
        for (double value : values) {
            const double diff = value - mean;
            squared += diff * diff;
        }
        return squared / static_cast<double>(values.size() - 1);
    }

    static double robust_variance(const std::vector<double> &values) {
        if (values.size() < 2) {
            return 0.0;
        }
        const double center = median(values);
        std::vector<double> deviations;
        deviations.reserve(values.size());
        for (double value : values) {
            deviations.push_back(std::abs(value - center));
        }
        const double mad = median(std::move(deviations));
        const double sigma = 1.4826 * mad;
        if (sigma > 0.0 && std::isfinite(sigma)) {
            return sigma * sigma;
        }
        return sample_variance(values);
    }

    static std::vector<double> difference(const std::vector<double> &values, int order = 1) {
        std::vector<double> out(values);
        for (int level = 0; level < order; ++level) {
            if (out.size() < 2) {
                out.clear();
                return out;
            }
            std::vector<double> next;
            next.reserve(out.size() - 1);
            for (std::size_t i = 1; i < out.size(); ++i) {
                next.push_back(out[i] - out[i - 1]);
            }
            out = std::move(next);
        }
        return out;
    }

    static KalmanMovingAverageTuning tune_vector(const std::vector<double> &prices, double dt, double min_variance) {
        if (prices.empty()) {
            throw nb::value_error("close must contain at least one finite value");
        }
        const double safe_dt = dt > 0.0 ? dt : 1.0;
        const double floor = min_variance > 0.0 ? min_variance : 1.0e-12;
        const std::vector<double> first_diff = difference(prices, 1);
        const std::vector<double> second_diff = difference(prices, 2);

        double measurement_variance = 0.0;
        if (second_diff.size() >= 2) {
            measurement_variance = robust_variance(second_diff) / 6.0;
        } else if (!first_diff.empty()) {
            measurement_variance = robust_variance(first_diff) / 2.0;
        }
        const double signal_variance = variance_floor(robust_variance(prices), floor);
        const double upper_measurement = std::max(signal_variance * 0.5, floor);
        measurement_variance = variance_floor(std::min(measurement_variance, upper_measurement), floor);

        std::vector<double> velocity;
        velocity.reserve(first_diff.size());
        for (double value : first_diff) {
            velocity.push_back(value / safe_dt);
        }

        const std::size_t velocity_seed_size = std::min<std::size_t>(8, velocity.size());
        const std::vector<double> velocity_seed(velocity.begin(), velocity.begin() + static_cast<std::ptrdiff_t>(velocity_seed_size));
        const double initial_velocity = velocity_seed.empty() ? 0.0 : median(velocity_seed);
        const double velocity_variance = variance_floor(
            robust_variance(velocity),
            std::max(measurement_variance / (safe_dt * safe_dt), floor));

        std::vector<double> acceleration;
        acceleration.reserve(second_diff.size());
        for (double value : second_diff) {
            acceleration.push_back(value / (safe_dt * safe_dt));
        }

        const double drive_variance = variance_floor(
            robust_variance(acceleration) - 6.0 * measurement_variance / (safe_dt * safe_dt * safe_dt * safe_dt),
            floor);
        const double process_position_variance = variance_floor(
            0.25 * drive_variance * safe_dt * safe_dt * safe_dt * safe_dt,
            std::max(measurement_variance * 0.01, floor));
        const double process_velocity_variance = variance_floor(
            drive_variance * safe_dt * safe_dt,
            std::max(velocity_variance * 0.001, floor));

        return KalmanMovingAverageTuning{
            prices.front(),
            initial_velocity,
            safe_dt,
            std::max(signal_variance, std::max(measurement_variance * 10.0, floor)),
            velocity_variance,
            process_position_variance,
            process_velocity_variance,
            measurement_variance,
        };
    }

    void initialize(double price) {
        filter_ = kalman::make_constant_velocity_1d(
            price,
            initial_velocity_,
            dt_,
            position_variance_,
            velocity_variance_,
            process_position_variance_,
            process_velocity_variance_,
            measurement_variance_);
        initialized_ = true;
        last_ = price;
    }

    double initial_price_;
    double initial_velocity_;
    double dt_;
    double position_variance_;
    double velocity_variance_;
    double process_position_variance_;
    double process_velocity_variance_;
    double measurement_variance_;
    kalman::LocalLinearTrendFilter filter_;
    bool initialized_;
    std::size_t count_;
    double last_;
    bool fillna_;
};

class KalmanInnovationZScore {
public:
    KalmanInnovationZScore(double initial_price = nan(),
                           double initial_velocity = 0.0,
                           double dt = 1.0,
                           double position_variance = 1.0,
                           double velocity_variance = 1.0,
                           double process_position_variance = 1.0e-4,
                           double process_velocity_variance = 1.0e-3,
                           double measurement_variance = 0.25,
                           bool fillna = true)
        : initial_price_(initial_price),
          initial_velocity_(initial_velocity),
          dt_(dt > 0.0 ? dt : 1.0),
          position_variance_(variance_floor(position_variance)),
          velocity_variance_(variance_floor(velocity_variance)),
          process_position_variance_(variance_floor(process_position_variance)),
          process_velocity_variance_(variance_floor(process_velocity_variance)),
          measurement_variance_(variance_floor(measurement_variance)),
          initialized_(false),
          count_(0),
          last_(nan()),
          fillna_(fillna) {}

    KalmanInnovationZScore(const KalmanInnovationZScoreTuning &tuning, bool fillna = true)
        : KalmanInnovationZScore(
              tuning.initial_price,
              tuning.initial_velocity,
              tuning.dt,
              tuning.position_variance,
              tuning.velocity_variance,
              tuning.process_position_variance,
              tuning.process_velocity_variance,
              tuning.measurement_variance,
              fillna) {}

    double update(double close) {
        if (!initialized_) {
            initialize(std::isfinite(initial_price_) ? initial_price_ : close);
            if (!std::isfinite(initial_price_)) {
                last_ = 0.0;
                ++count_;
                return (!fillna_ && count_ < 2) ? nan() : last_;
            }
        }

        const auto stats = filter_.update(kalman::Vec<1>{close});
        const double innovation_variance = stats.S(0, 0);
        if (!stats.ok || !std::isfinite(innovation_variance) || innovation_variance <= 0.0) {
            last_ = nan();
        } else {
            last_ = stats.innovation[0] / std::sqrt(innovation_variance);
        }
        ++count_;
        return (!fillna_ && count_ < 2) ? nan() : last_;
    }

    void advance(double close) {
        (void) update(close);
    }

    double last() const {
        return last_;
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(values[i]));
        }
        return make_array(std::move(output));
    }

    template <typename Array>
    static KalmanInnovationZScoreTuning tune_array(const Array &close, double dt = 1.0, double min_variance = 1.0e-12) {
        return from_moving_average_tuning(KalmanMovingAverage::tune_array(close, dt, min_variance));
    }

    static KalmanInnovationZScoreTuning tune_records(nb::iterable records, double dt = 1.0, double min_variance = 1.0e-12) {
        return from_moving_average_tuning(KalmanMovingAverage::tune_records(records, dt, min_variance));
    }

private:
    static double variance_floor(double value, double minimum = kalman::kDefaultVarianceFloor) {
        if (!std::isfinite(value) || value < minimum) {
            return minimum;
        }
        return value;
    }

    static KalmanInnovationZScoreTuning from_moving_average_tuning(const KalmanMovingAverageTuning &tuning) {
        return KalmanInnovationZScoreTuning{
            tuning.initial_price,
            tuning.initial_velocity,
            tuning.dt,
            tuning.position_variance,
            tuning.velocity_variance,
            tuning.process_position_variance,
            tuning.process_velocity_variance,
            tuning.measurement_variance,
        };
    }

    void initialize(double price) {
        filter_ = kalman::make_constant_velocity_1d(
            price,
            initial_velocity_,
            dt_,
            position_variance_,
            velocity_variance_,
            process_position_variance_,
            process_velocity_variance_,
            measurement_variance_);
        initialized_ = true;
        last_ = 0.0;
    }

    double initial_price_;
    double initial_velocity_;
    double dt_;
    double position_variance_;
    double velocity_variance_;
    double process_position_variance_;
    double process_velocity_variance_;
    double measurement_variance_;
    kalman::LocalLinearTrendFilter filter_;
    bool initialized_;
    std::size_t count_;
    double last_;
    bool fillna_;
};

class KalmanPredictionBands {
public:
    KalmanPredictionBands(double initial_price = nan(),
                          double initial_velocity = 0.0,
                          double dt = 1.0,
                          double position_variance = 1.0,
                          double velocity_variance = 1.0,
                          double process_position_variance = 1.0e-4,
                          double process_velocity_variance = 1.0e-3,
                          double measurement_variance = 0.25,
                          double multiplier = 2.0,
                          bool fillna = true)
        : initial_price_(initial_price),
          initial_velocity_(initial_velocity),
          dt_(dt > 0.0 ? dt : 1.0),
          position_variance_(variance_floor(position_variance)),
          velocity_variance_(variance_floor(velocity_variance)),
          process_position_variance_(variance_floor(process_position_variance)),
          process_velocity_variance_(variance_floor(process_velocity_variance)),
          measurement_variance_(variance_floor(measurement_variance)),
          multiplier_(std::isfinite(multiplier) ? std::abs(multiplier) : 2.0),
          initialized_(false),
          count_(0),
          last_{nan(), nan(), nan()},
          fillna_(fillna) {}

    KalmanPredictionBands(const KalmanPredictionBandsTuning &tuning, double multiplier = 2.0, bool fillna = true)
        : KalmanPredictionBands(
              tuning.initial_price,
              tuning.initial_velocity,
              tuning.dt,
              tuning.position_variance,
              tuning.velocity_variance,
              tuning.process_position_variance,
              tuning.process_velocity_variance,
              tuning.measurement_variance,
              multiplier,
              fillna) {}

    KalmanPredictionBandsResult update(double close) {
        if (!initialized_) {
            initialize(std::isfinite(initial_price_) ? initial_price_ : close);
            if (!std::isfinite(initial_price_)) {
                const double band = multiplier_ * std::sqrt(measurement_variance_);
                last_ = KalmanPredictionBandsResult{close, close + band, close - band};
                ++count_;
                return output_for_warmup();
            }
        }

        const auto stats = filter_.update(kalman::Vec<1>{close});
        const double innovation_variance = stats.S(0, 0);
        if (!stats.ok || !std::isfinite(innovation_variance) || innovation_variance <= 0.0) {
            last_ = KalmanPredictionBandsResult{nan(), nan(), nan()};
        } else {
            const double middle = close - stats.innovation[0];
            const double band = multiplier_ * std::sqrt(innovation_variance);
            last_ = KalmanPredictionBandsResult{middle, middle + band, middle - band};
        }
        ++count_;
        return output_for_warmup();
    }

    void advance(double close) {
        (void) update(close);
    }

    const KalmanPredictionBandsResult &last() const {
        return last_;
    }

    template <typename Array>
    KalmanPredictionBandsBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> middle(size);
        std::vector<double> upper(size);
        std::vector<double> lower(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const KalmanPredictionBandsResult out = update(static_cast<double>(values[i]));
            middle[i] = out.middle;
            upper[i] = out.upper;
            lower[i] = out.lower;
        }
        return KalmanPredictionBandsBatchResult{
            make_array(std::move(middle)),
            make_array(std::move(upper)),
            make_array(std::move(lower)),
        };
    }

    template <typename Array>
    static KalmanPredictionBandsTuning tune_array(const Array &close, double dt = 1.0, double min_variance = 1.0e-12) {
        return from_moving_average_tuning(KalmanMovingAverage::tune_array(close, dt, min_variance));
    }

    static KalmanPredictionBandsTuning tune_records(nb::iterable records, double dt = 1.0, double min_variance = 1.0e-12) {
        return from_moving_average_tuning(KalmanMovingAverage::tune_records(records, dt, min_variance));
    }

private:
    static double variance_floor(double value, double minimum = kalman::kDefaultVarianceFloor) {
        if (!std::isfinite(value) || value < minimum) {
            return minimum;
        }
        return value;
    }

    static KalmanPredictionBandsTuning from_moving_average_tuning(const KalmanMovingAverageTuning &tuning) {
        return KalmanPredictionBandsTuning{
            tuning.initial_price,
            tuning.initial_velocity,
            tuning.dt,
            tuning.position_variance,
            tuning.velocity_variance,
            tuning.process_position_variance,
            tuning.process_velocity_variance,
            tuning.measurement_variance,
        };
    }

    KalmanPredictionBandsResult output_for_warmup() const {
        if (!fillna_ && count_ < 2) {
            return KalmanPredictionBandsResult{nan(), nan(), nan()};
        }
        return last_;
    }

    void initialize(double price) {
        filter_ = kalman::make_constant_velocity_1d(
            price,
            initial_velocity_,
            dt_,
            position_variance_,
            velocity_variance_,
            process_position_variance_,
            process_velocity_variance_,
            measurement_variance_);
        initialized_ = true;
        const double band = multiplier_ * std::sqrt(measurement_variance_);
        last_ = KalmanPredictionBandsResult{price, price + band, price - band};
    }

    double initial_price_;
    double initial_velocity_;
    double dt_;
    double position_variance_;
    double velocity_variance_;
    double process_position_variance_;
    double process_velocity_variance_;
    double measurement_variance_;
    double multiplier_;
    kalman::LocalLinearTrendFilter filter_;
    bool initialized_;
    std::size_t count_;
    KalmanPredictionBandsResult last_;
    bool fillna_;
};

class KalmanTrendSignal {
public:
    KalmanTrendSignal(double initial_price = nan(),
                      double initial_velocity = 0.0,
                      double dt = 1.0,
                      double position_variance = 1.0,
                      double velocity_variance = 1.0,
                      double process_position_variance = 1.0e-4,
                      double process_velocity_variance = 1.0e-3,
                      double measurement_variance = 0.25,
                      bool fillna = true)
        : initial_price_(initial_price),
          initial_velocity_(initial_velocity),
          dt_(dt > 0.0 ? dt : 1.0),
          position_variance_(variance_floor(position_variance)),
          velocity_variance_(variance_floor(velocity_variance)),
          process_position_variance_(variance_floor(process_position_variance)),
          process_velocity_variance_(variance_floor(process_velocity_variance)),
          measurement_variance_(variance_floor(measurement_variance)),
          initialized_(false),
          count_(0),
          last_{nan(), nan()},
          fillna_(fillna) {}

    KalmanTrendSignal(const KalmanTrendSignalTuning &tuning, bool fillna = true)
        : KalmanTrendSignal(
              tuning.initial_price,
              tuning.initial_velocity,
              tuning.dt,
              tuning.position_variance,
              tuning.velocity_variance,
              tuning.process_position_variance,
              tuning.process_velocity_variance,
              tuning.measurement_variance,
              fillna) {}

    KalmanTrendSignalResult update(double close) {
        if (!initialized_) {
            initialize(std::isfinite(initial_price_) ? initial_price_ : close);
            if (!std::isfinite(initial_price_)) {
                last_ = KalmanTrendSignalResult{close, 0.0};
                ++count_;
                return output_for_warmup();
            }
        }

        (void) filter_.update(kalman::Vec<1>{close});
        const double trend = filter_.x[0];
        last_ = KalmanTrendSignalResult{trend, signal_for(close, trend)};
        ++count_;
        return output_for_warmup();
    }

    void advance(double close) {
        (void) update(close);
    }

    const KalmanTrendSignalResult &last() const {
        return last_;
    }

    template <typename Array>
    KalmanTrendSignalBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> trend(size);
        std::vector<double> signal(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const KalmanTrendSignalResult out = update(static_cast<double>(values[i]));
            trend[i] = out.trend;
            signal[i] = out.signal;
        }
        return KalmanTrendSignalBatchResult{
            make_array(std::move(trend)),
            make_array(std::move(signal)),
        };
    }

    template <typename Array>
    static KalmanTrendSignalTuning tune_array(const Array &close, double dt = 1.0, double min_variance = 1.0e-12) {
        return from_moving_average_tuning(KalmanMovingAverage::tune_array(close, dt, min_variance));
    }

    static KalmanTrendSignalTuning tune_records(nb::iterable records, double dt = 1.0, double min_variance = 1.0e-12) {
        return from_moving_average_tuning(KalmanMovingAverage::tune_records(records, dt, min_variance));
    }

private:
    static double variance_floor(double value, double minimum = kalman::kDefaultVarianceFloor) {
        if (!std::isfinite(value) || value < minimum) {
            return minimum;
        }
        return value;
    }

    static double signal_for(double close, double trend) {
        if (!std::isfinite(close) || !std::isfinite(trend)) {
            return nan();
        }
        if (close > trend) {
            return 1.0;
        }
        if (close < trend) {
            return -1.0;
        }
        return 0.0;
    }

    static KalmanTrendSignalTuning from_moving_average_tuning(const KalmanMovingAverageTuning &tuning) {
        return KalmanTrendSignalTuning{
            tuning.initial_price,
            tuning.initial_velocity,
            tuning.dt,
            tuning.position_variance,
            tuning.velocity_variance,
            tuning.process_position_variance,
            tuning.process_velocity_variance,
            tuning.measurement_variance,
        };
    }

    KalmanTrendSignalResult output_for_warmup() const {
        if (!fillna_ && count_ < 2) {
            return KalmanTrendSignalResult{nan(), nan()};
        }
        return last_;
    }

    void initialize(double price) {
        filter_ = kalman::make_constant_velocity_1d(
            price,
            initial_velocity_,
            dt_,
            position_variance_,
            velocity_variance_,
            process_position_variance_,
            process_velocity_variance_,
            measurement_variance_);
        initialized_ = true;
        last_ = KalmanTrendSignalResult{price, 0.0};
    }

    double initial_price_;
    double initial_velocity_;
    double dt_;
    double position_variance_;
    double velocity_variance_;
    double process_position_variance_;
    double process_velocity_variance_;
    double measurement_variance_;
    kalman::LocalLinearTrendFilter filter_;
    bool initialized_;
    std::size_t count_;
    KalmanTrendSignalResult last_;
    bool fillna_;
};

class KalmanVelocityOscillator {
public:
    KalmanVelocityOscillator(double initial_price = nan(),
                             double initial_velocity = 0.0,
                             double dt = 1.0,
                             double position_variance = 1.0,
                             double velocity_variance = 1.0,
                             double process_position_variance = 1.0e-4,
                             double process_velocity_variance = 1.0e-3,
                             double measurement_variance = 0.25,
                             bool fillna = true)
        : initial_price_(initial_price),
          initial_velocity_(initial_velocity),
          dt_(dt > 0.0 ? dt : 1.0),
          position_variance_(variance_floor(position_variance)),
          velocity_variance_(variance_floor(velocity_variance)),
          process_position_variance_(variance_floor(process_position_variance)),
          process_velocity_variance_(variance_floor(process_velocity_variance)),
          measurement_variance_(variance_floor(measurement_variance)),
          initialized_(false),
          count_(0),
          last_(nan()),
          fillna_(fillna) {}

    KalmanVelocityOscillator(const KalmanVelocityOscillatorTuning &tuning, bool fillna = true)
        : KalmanVelocityOscillator(
              tuning.initial_price,
              tuning.initial_velocity,
              tuning.dt,
              tuning.position_variance,
              tuning.velocity_variance,
              tuning.process_position_variance,
              tuning.process_velocity_variance,
              tuning.measurement_variance,
              fillna) {}

    double update(double close) {
        if (!initialized_) {
            initialize(std::isfinite(initial_price_) ? initial_price_ : close);
            if (!std::isfinite(initial_price_)) {
                last_ = initial_velocity_;
                ++count_;
                return (!fillna_ && count_ < 2) ? nan() : last_;
            }
        }

        (void) filter_.update(kalman::Vec<1>{close});
        last_ = filter_.x[1];
        ++count_;
        return (!fillna_ && count_ < 2) ? nan() : last_;
    }

    void advance(double close) {
        (void) update(close);
    }

    double last() const {
        return last_;
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(values[i]));
        }
        return make_array(std::move(output));
    }

    template <typename Array>
    static KalmanVelocityOscillatorTuning tune_array(const Array &close, double dt = 1.0, double min_variance = 1.0e-12) {
        return from_moving_average_tuning(KalmanMovingAverage::tune_array(close, dt, min_variance));
    }

    static KalmanVelocityOscillatorTuning tune_records(nb::iterable records, double dt = 1.0, double min_variance = 1.0e-12) {
        return from_moving_average_tuning(KalmanMovingAverage::tune_records(records, dt, min_variance));
    }

private:
    static double variance_floor(double value, double minimum = kalman::kDefaultVarianceFloor) {
        if (!std::isfinite(value) || value < minimum) {
            return minimum;
        }
        return value;
    }

    static KalmanVelocityOscillatorTuning from_moving_average_tuning(const KalmanMovingAverageTuning &tuning) {
        return KalmanVelocityOscillatorTuning{
            tuning.initial_price,
            tuning.initial_velocity,
            tuning.dt,
            tuning.position_variance,
            tuning.velocity_variance,
            tuning.process_position_variance,
            tuning.process_velocity_variance,
            tuning.measurement_variance,
        };
    }

    void initialize(double price) {
        filter_ = kalman::make_constant_velocity_1d(
            price,
            initial_velocity_,
            dt_,
            position_variance_,
            velocity_variance_,
            process_position_variance_,
            process_velocity_variance_,
            measurement_variance_);
        initialized_ = true;
        last_ = initial_velocity_;
    }

    double initial_price_;
    double initial_velocity_;
    double dt_;
    double position_variance_;
    double velocity_variance_;
    double process_position_variance_;
    double process_velocity_variance_;
    double measurement_variance_;
    kalman::LocalLinearTrendFilter filter_;
    bool initialized_;
    std::size_t count_;
    double last_;
    bool fillna_;
};

class KalmanLocalLinearTrend {
public:
    KalmanLocalLinearTrend(double initial_level = nan(),
                           double initial_trend = 0.0,
                           double dt = 1.0,
                           double level_variance = 1.0,
                           double trend_variance = 1.0,
                           double process_level_variance = 1.0e-4,
                           double process_trend_variance = 1.0e-3,
                           double observation_variance = 0.25,
                           bool fillna = true)
        : initial_level_(initial_level),
          initial_trend_(initial_trend),
          dt_(dt > 0.0 ? dt : 1.0),
          level_variance_(variance_floor(level_variance)),
          trend_variance_(variance_floor(trend_variance)),
          process_level_variance_(variance_floor(process_level_variance)),
          process_trend_variance_(variance_floor(process_trend_variance)),
          observation_variance_(variance_floor(observation_variance)),
          initialized_(false),
          count_(0),
          last_{nan(), nan()},
          fillna_(fillna) {}

    KalmanLocalLinearTrend(const KalmanLocalLinearTrendTuning &tuning, bool fillna = true)
        : KalmanLocalLinearTrend(
              tuning.initial_level,
              tuning.initial_trend,
              tuning.dt,
              tuning.level_variance,
              tuning.trend_variance,
              tuning.process_level_variance,
              tuning.process_trend_variance,
              tuning.observation_variance,
              fillna) {}

    KalmanLocalLinearTrendResult update(double close) {
        if (!initialized_) {
            initialize(std::isfinite(initial_level_) ? initial_level_ : close);
            if (!std::isfinite(initial_level_)) {
                last_ = KalmanLocalLinearTrendResult{close, initial_trend_};
                ++count_;
                return output_for_warmup();
            }
        }

        (void) filter_.update(kalman::Vec<1>{close});
        last_ = KalmanLocalLinearTrendResult{filter_.x[0], filter_.x[1]};
        ++count_;
        return output_for_warmup();
    }

    void advance(double close) {
        (void) update(close);
    }

    const KalmanLocalLinearTrendResult &last() const {
        return last_;
    }

    template <typename Array>
    KalmanLocalLinearTrendBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> level(size);
        std::vector<double> trend(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const KalmanLocalLinearTrendResult out = update(static_cast<double>(values[i]));
            level[i] = out.level;
            trend[i] = out.trend;
        }
        return KalmanLocalLinearTrendBatchResult{
            make_array(std::move(level)),
            make_array(std::move(trend)),
        };
    }

    template <typename Array>
    static KalmanLocalLinearTrendTuning tune_array(const Array &close, double dt = 1.0, double min_variance = 1.0e-12) {
        return from_moving_average_tuning(KalmanMovingAverage::tune_array(close, dt, min_variance));
    }

    static KalmanLocalLinearTrendTuning tune_records(nb::iterable records, double dt = 1.0, double min_variance = 1.0e-12) {
        return from_moving_average_tuning(KalmanMovingAverage::tune_records(records, dt, min_variance));
    }

private:
    static double variance_floor(double value, double minimum = kalman::kDefaultVarianceFloor) {
        if (!std::isfinite(value) || value < minimum) {
            return minimum;
        }
        return value;
    }

    static KalmanLocalLinearTrendTuning from_moving_average_tuning(const KalmanMovingAverageTuning &tuning) {
        return KalmanLocalLinearTrendTuning{
            tuning.initial_price,
            tuning.initial_velocity,
            tuning.dt,
            tuning.position_variance,
            tuning.velocity_variance,
            tuning.process_position_variance,
            tuning.process_velocity_variance,
            tuning.measurement_variance,
        };
    }

    KalmanLocalLinearTrendResult output_for_warmup() const {
        if (!fillna_ && count_ < 2) {
            return KalmanLocalLinearTrendResult{nan(), nan()};
        }
        return last_;
    }

    void initialize(double level) {
        filter_ = kalman::make_constant_velocity_1d(
            level,
            initial_trend_,
            dt_,
            level_variance_,
            trend_variance_,
            process_level_variance_,
            process_trend_variance_,
            observation_variance_);
        initialized_ = true;
        last_ = KalmanLocalLinearTrendResult{level, initial_trend_};
    }

    double initial_level_;
    double initial_trend_;
    double dt_;
    double level_variance_;
    double trend_variance_;
    double process_level_variance_;
    double process_trend_variance_;
    double observation_variance_;
    kalman::LocalLinearTrendFilter filter_;
    bool initialized_;
    std::size_t count_;
    KalmanLocalLinearTrendResult last_;
    bool fillna_;
};

class KeltnerChannel {
public:
    KeltnerChannel(double span = 20.0, double window_atr = 20.0, bool fillna = false, double multiplier = 2.0)
        : middle_(nb::none(), nb::float_(span), nb::none(), false),
          atr_(window_atr, true),
          multiplier_(multiplier),
          start_(true),
          last_{nan(), nan(), nan()} {
        static_cast<void>(fillna);
    }

    KeltnerChannelResult update(double close, double high, double low) {
        update_core(close, high, low);
        return last_;
    }

    void advance(double close, double high, double low) {
        update_core(close, high, low);
    }

    inline const KeltnerChannelResult &last() const {
        return last_;
    }

private:
    inline void update_core(double close, double high, double low) {
        const double atr = atr_.update(close, high, low);
        const double tp = middle_.update(close);
        last_ = {tp, tp + multiplier_ * atr, tp - multiplier_ * atr};
    }

    EWMA middle_;
    ATR atr_;
    double multiplier_;
    bool start_;
    KeltnerChannelResult last_;
};

class KeltnerChannelOriginal {
public:
    explicit KeltnerChannelOriginal(int window = 20, bool fillna = false)
        : window_(window),
          fillna_(fillna),
          multiplier_(2.0),
          high_(window, fillna),
          middle_(window, fillna),
          low_(window, fillna),
          last_{nan(), nan(), nan()} {}

    KeltnerChannelResult update(double close, double high, double low) {
        update_core(close, high, low);
        return last_;
    }

    void advance(double close, double high, double low) {
        update_core(close, high, low);
    }

    inline const KeltnerChannelResult &last() const {
        return last_;
    }

private:
    inline void update_core(double close, double high, double low) {
        last_ = {
            middle_.update((high + close + low) / 3.0),
            high_.update((high * 4.0 + close - 2.0 * low) / 3.0),
            low_.update((close + 4.0 * low - 2.0 * high) / 3.0),
        };
    }

    int window_;
    bool fillna_;
    double multiplier_;
    SMA high_;
    SMA middle_;
    SMA low_;
    KeltnerChannelResult last_;
};

class MACD {
public:
    MACD(int a = 12, int b = 26, int c = 9, bool fillna = false)
        : a_(a, true),
          b_(b, true),
          c_(c, fillna),
          counter_(0),
          fillna_(fillna),
          window_(std::max(a, b) + c) {}

    double update(double value) {
        const double retval = c_.update(a_.update(value) - b_.update(value));
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : retval;
    }

    nb::object batch(const InputArray &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(input(i));
        }
        return make_array(std::move(output));
    }

private:
    EMA a_;
    EMA b_;
    EMA c_;
    long counter_;
    bool fillna_;
    int window_;
};

class MassIndex {
public:
    MassIndex(int single = 9, int double_window = 9, int summation = 25, bool fillna = false)
        : single_(single, true),
          double_(double_window, true),
          summation_(summation, true),
          counter_(0),
          window_(std::max(single, double_window) + summation),
          fillna_(fillna) {}

    double update(double high, double low) {
        if (high < low) {
            std::swap(high, low);
        }

        const double single = single_.update(high - low);
        const double double_value = double_.update(single);
        const double retval = summation_.update(single / double_value);
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : retval;
    }

private:
    EMA single_;
    EMA double_;
    Summation summation_;
    long counter_;
    int window_;
    bool fillna_;
};

class AccumulationDistribution {
public:
    AccumulationDistribution()
        : value_(0.0) {}

    double update(double close, double high, double low, double volume) {
        value_ += money_flow_multiplier(close, high, low) * volume;
        return value_;
    }

private:
    double value_;
};

class ChaikinOscillator {
public:
    ChaikinOscillator(int fast = 3, int slow = 10)
        : ad_(),
          fast_(fast, true),
          slow_(slow, true) {}

    double update(double close, double high, double low, double volume) {
        const double ad = ad_.update(close, high, low, volume);
        return fast_.update(ad) - slow_.update(ad);
    }

private:
    AccumulationDistribution ad_;
    EMA fast_;
    EMA slow_;
};

class AveragePrice {
public:
    double update(double open, double high, double low, double close) {
        return (open + high + low + close) * 0.25;
    }
};

class MedianPrice {
public:
    double update(double high, double low) {
        return (high + low) * 0.5;
    }
};

class TypicalPrice {
public:
    double update(double close, double high, double low) {
        return (high + low + close) / 3.0;
    }
};

class WeightedClosePrice {
public:
    double update(double close, double high, double low) {
        return (high + low + 2.0 * close) * 0.25;
    }
};

class BalanceOfPower {
public:
    double update(double open, double high, double low, double close) {
        return safe_divide(close - open, high - low);
    }
};

class TrueRange {
public:
    TrueRange()
        : previous_close_(0.0),
          first_(true) {}

    double update(double close, double high, double low) {
        const double retval = first_ ? high - low : true_range(close, high, low, previous_close_);
        previous_close_ = close;
        first_ = false;
        return retval;
    }

    template <typename Array0, typename Array1, typename Array2>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        batch_kernels::true_range(
            close.data(),
            high.data(),
            low.data(),
            size,
            previous_close_,
            first_,
            output.data());
        return make_array(std::move(output));
    }

private:
    double previous_close_;
    bool first_;
};

class ChoppinessIndex {
public:
    ChoppinessIndex(int window = 14, bool fillna = true)
        : window_(std::max(window, 1)),
          true_ranges_(window_),
          highs_(window_, true),
          lows_(window_, false),
          previous_close_(0.0),
          first_(true),
          fillna_(fillna) {}

    double update(double close, double high, double low) {
        const double tr = first_ ? high - low : true_range(close, high, low, previous_close_);
        previous_close_ = close;
        first_ = false;

        true_ranges_.push(tr);
        highs_.push(high);
        lows_.push(low);

        if (!fillna_ && !true_ranges_.full()) {
            return nan();
        }

        const double range = highs_.value() - lows_.value();
        const double tr_sum = true_ranges_.sum();
        if (range <= 0.0 || tr_sum <= 0.0) {
            return 100.0;
        }

        const std::size_t periods = fillna_
            ? std::max<std::size_t>(true_ranges_.size(), 2)
            : static_cast<std::size_t>(window_);
        return 100.0 * std::log10(tr_sum / range) / std::log10(static_cast<double>(periods));
    }

    template <typename Array0, typename Array1, typename Array2>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        const auto *close_values = close.data();
        const auto *high_values = high.data();
        const auto *low_values = low.data();

        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(
                static_cast<double>(close_values[i]),
                static_cast<double>(high_values[i]),
                static_cast<double>(low_values[i]));
        }

        return make_array(std::move(output));
    }

private:
    int window_;
    RollingSumWindow true_ranges_;
    RollingExtreme highs_;
    RollingExtreme lows_;
    double previous_close_;
    bool first_;
    bool fillna_;
};

class NormalizedATR {
public:
    NormalizedATR(double window = 14.0, bool fillna = true)
        : atr_(window, fillna) {}

    double update(double close, double high, double low) {
        return 100.0 * atr_.update(close, high, low) / close;
    }

    template <typename Array0, typename Array1, typename Array2>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        atr_.batch_to(close, high, low, output.data());
        const auto *close_values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = 100.0 * output[i] / static_cast<double>(close_values[i]);
        }
        return make_array(std::move(output));
    }

private:
    ATR atr_;
};

class OnBalanceVolume {
public:
    OnBalanceVolume()
        : value_(0.0),
          previous_close_(0.0),
          first_(true) {}

    double update(double close, double volume) {
        if (first_) {
            value_ = volume;
            first_ = false;
        } else if (close > previous_close_) {
            value_ += volume;
        } else if (close < previous_close_) {
            value_ -= volume;
        }

        previous_close_ = close;
        return value_;
    }

private:
    double value_;
    double previous_close_;
    bool first_;
};

class OrderFlowImbalance {
public:
    OrderFlowImbalance(int window = 1, bool fillna = true)
        : events_(window),
          has_previous_(false),
          previous_bid_price_(0.0),
          previous_bid_size_(0.0),
          previous_ask_price_(0.0),
          previous_ask_size_(0.0),
          fillna_(fillna),
          last_(fillna ? 0.0 : nan()) {}

    inline double update(double bid_price, double bid_size, double ask_price, double ask_size) {
        double event = 0.0;
        if (has_previous_) {
            if (bid_price >= previous_bid_price_) {
                event += bid_size;
            }
            if (bid_price <= previous_bid_price_) {
                event -= previous_bid_size_;
            }
            if (ask_price <= previous_ask_price_) {
                event -= ask_size;
            }
            if (ask_price >= previous_ask_price_) {
                event += previous_ask_size_;
            }
        }

        events_.push(event);
        previous_bid_price_ = bid_price;
        previous_bid_size_ = bid_size;
        previous_ask_price_ = ask_price;
        previous_ask_size_ = ask_size;
        has_previous_ = true;
        last_ = (!fillna_ && !events_.full()) ? nan() : events_.sum();
        return last_;
    }

    inline void advance(double bid_price, double bid_size, double ask_price, double ask_size) {
        (void) update(bid_price, bid_size, ask_price, ask_size);
    }

    inline double last() const {
        return last_;
    }

    template <typename Array0, typename Array1, typename Array2, typename Array3>
    nb::object batch_array(const Array0 &bid_price, const Array1 &bid_size, const Array2 &ask_price, const Array3 &ask_size) {
        return batch_update4(*this, bid_price, bid_size, ask_price, ask_size);
    }

private:
    RollingSumWindow events_;
    bool has_previous_;
    double previous_bid_price_;
    double previous_bid_size_;
    double previous_ask_price_;
    double previous_ask_size_;
    bool fillna_;
    double last_;
};

class VPIN {
public:
    VPIN(double bucket_volume = 1000000.0, int buckets = 50, int price_change_window = 50, bool fillna = true)
        : bucket_volume_(checked_bucket_volume(bucket_volume)),
          imbalances_(checked_window(buckets, "buckets")),
          price_changes_(checked_window(price_change_window, "price_change_window")),
          sum_price_change_(0.0),
          sum_price_change_squared_(0.0),
          partial_buy_volume_(0.0),
          partial_sell_volume_(0.0),
          partial_volume_(0.0),
          previous_close_(0.0),
          has_previous_(false),
          fillna_(fillna),
          last_(fillna ? 0.0 : nan()) {}

    inline double update(double close, double volume) {
        const double price_change = has_previous_ ? close - previous_close_ : 0.0;
        update_price_change(price_change);

        const double buy_fraction = classify_buy_fraction(price_change);
        const double positive_volume = std::max(volume, 0.0);
        add_classified_volume(positive_volume * buy_fraction, positive_volume * (1.0 - buy_fraction));

        previous_close_ = close;
        has_previous_ = true;
        last_ = current_value();
        return last_;
    }

    inline void advance(double close, double volume) {
        (void) update(close, volume);
    }

    inline double last() const {
        return last_;
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &close, const Array1 &volume) {
        const std::size_t size = close.shape(0);
        require_same_size(size, volume.shape(0));
        std::vector<double> output(size);
        const auto *close_values = close.data();
        const auto *volume_values = volume.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(close_values[i]), static_cast<double>(volume_values[i]));
        }
        return make_array(std::move(output));
    }

private:
    static double checked_bucket_volume(double bucket_volume) {
        if (bucket_volume <= 0.0) {
            throw nb::value_error("bucket_volume must be positive");
        }
        return bucket_volume;
    }

    static int checked_window(int window, const char *name) {
        if (window <= 0) {
            throw nb::value_error((std::string(name) + " must be positive").c_str());
        }
        return window;
    }

    inline void update_price_change(double price_change) {
        if (price_changes_.full()) {
            const double old = price_changes_.oldest();
            sum_price_change_ -= old;
            sum_price_change_squared_ -= old * old;
        }
        price_changes_.push(price_change);
        sum_price_change_ += price_change;
        sum_price_change_squared_ += price_change * price_change;
    }

    inline double price_change_stddev() const {
        const std::size_t n = price_changes_.size();
        if (n < 2) {
            return 0.0;
        }
        const double n_value = static_cast<double>(n);
        const double variance = (sum_price_change_squared_ - (sum_price_change_ * sum_price_change_) / n_value) / (n_value - 1.0);
        return variance <= 0.0 ? 0.0 : std::sqrt(variance);
    }

    inline double classify_buy_fraction(double price_change) const {
        const double sigma = price_change_stddev();
        if (sigma == 0.0) {
            if (price_change > 0.0) {
                return 1.0;
            }
            if (price_change < 0.0) {
                return 0.0;
            }
            return 0.5;
        }
        return std::clamp(normal_cdf(price_change / sigma), 0.0, 1.0);
    }

    inline void complete_bucket() {
        imbalances_.push(std::abs(partial_buy_volume_ - partial_sell_volume_));
        partial_buy_volume_ = 0.0;
        partial_sell_volume_ = 0.0;
        partial_volume_ = 0.0;
    }

    inline void add_classified_volume(double buy_volume, double sell_volume) {
        double remaining_buy = buy_volume;
        double remaining_sell = sell_volume;
        double remaining_volume = buy_volume + sell_volume;
        while (remaining_volume > 0.0) {
            const double bucket_remaining = bucket_volume_ - partial_volume_;
            const double take = std::min(bucket_remaining, remaining_volume);
            const double fraction = safe_divide(take, remaining_volume);
            partial_buy_volume_ += remaining_buy * fraction;
            partial_sell_volume_ += remaining_sell * fraction;
            partial_volume_ += take;
            remaining_buy -= remaining_buy * fraction;
            remaining_sell -= remaining_sell * fraction;
            remaining_volume -= take;

            if (partial_volume_ >= bucket_volume_ - 1e-12) {
                complete_bucket();
            }
        }
    }

    inline double current_value() const {
        if (imbalances_.size() == 0) {
            return fillna_ ? 0.0 : nan();
        }
        if (!fillna_ && !imbalances_.full()) {
            return nan();
        }
        return safe_divide(imbalances_.sum(), bucket_volume_ * static_cast<double>(imbalances_.size()));
    }

    double bucket_volume_;
    RollingSumWindow imbalances_;
    RollingBuffer price_changes_;
    double sum_price_change_;
    double sum_price_change_squared_;
    double partial_buy_volume_;
    double partial_sell_volume_;
    double partial_volume_;
    double previous_close_;
    bool has_previous_;
    bool fillna_;
    double last_;
};

class KyleLambda {
public:
    KyleLambda(int window = 30, double scale = 1000000.0, bool fillna = true)
        : returns_(window),
          signed_sqrt_dollar_volume_(window),
          sum_return_(0.0),
          sum_flow_(0.0),
          sum_return_flow_(0.0),
          sum_flow_squared_(0.0),
          scale_(scale),
          previous_close_(0.0),
          has_previous_(false),
          fillna_(fillna),
          last_(fillna ? 0.0 : nan()) {}

    inline double update(double close, double signed_dollar_volume) {
        const double price_return = has_previous_ ? safe_divide(close - previous_close_, previous_close_) : 0.0;
        const double signed_sqrt_dollar_volume = signed_dollar_volume < 0.0
            ? -std::sqrt(-signed_dollar_volume)
            : std::sqrt(signed_dollar_volume);

        if (returns_.full()) {
            const double old_return = returns_.oldest();
            const double old_flow = signed_sqrt_dollar_volume_.oldest();
            sum_return_ -= old_return;
            sum_flow_ -= old_flow;
            sum_return_flow_ -= old_return * old_flow;
            sum_flow_squared_ -= old_flow * old_flow;
        }
        returns_.push(price_return);
        signed_sqrt_dollar_volume_.push(signed_sqrt_dollar_volume);
        sum_return_ += price_return;
        sum_flow_ += signed_sqrt_dollar_volume;
        sum_return_flow_ += price_return * signed_sqrt_dollar_volume;
        sum_flow_squared_ += signed_sqrt_dollar_volume * signed_sqrt_dollar_volume;
        previous_close_ = close;
        has_previous_ = true;
        last_ = (!fillna_ && !returns_.full()) ? nan() : slope() * scale_;
        return last_;
    }

    inline void advance(double close, double signed_dollar_volume) {
        (void) update(close, signed_dollar_volume);
    }

    inline double last() const {
        return last_;
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &close, const Array1 &signed_dollar_volume) {
        const std::size_t size = close.shape(0);
        require_same_size(size, signed_dollar_volume.shape(0));
        std::vector<double> output(size);
        const auto *close_values = close.data();
        const auto *signed_dollar_volume_values = signed_dollar_volume.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(
                static_cast<double>(close_values[i]),
                static_cast<double>(signed_dollar_volume_values[i]));
        }
        return make_array(std::move(output));
    }

private:
    inline double slope() const {
        const double n = static_cast<double>(returns_.size());
        const double covariance = n * sum_return_flow_ - sum_return_ * sum_flow_;
        const double variance = n * sum_flow_squared_ - sum_flow_ * sum_flow_;
        return safe_divide(covariance, variance);
    }

    RollingBuffer returns_;
    RollingBuffer signed_sqrt_dollar_volume_;
    double sum_return_;
    double sum_flow_;
    double sum_return_flow_;
    double sum_flow_squared_;
    double scale_;
    double previous_close_;
    bool has_previous_;
    bool fillna_;
    double last_;
};

class AmihudIlliquidity {
public:
    AmihudIlliquidity(int window = 30, double scale = 1000000.0, bool fillna = true)
        : ratios_(window),
          scale_(scale),
          previous_close_(0.0),
          has_previous_(false),
          fillna_(fillna),
          last_(fillna ? 0.0 : nan()) {}

    inline double update(double close, double volume) {
        const double price_return = has_previous_ ? safe_divide(close - previous_close_, previous_close_) : 0.0;
        const double dollar_volume = std::abs(close * volume);
        const double ratio = safe_divide(std::abs(price_return), dollar_volume) * scale_;
        ratios_.push(ratio);
        previous_close_ = close;
        has_previous_ = true;
        last_ = (!fillna_ && !ratios_.full()) ? nan() : safe_divide(ratios_.sum(), static_cast<double>(ratios_.size()));
        return last_;
    }

    inline void advance(double close, double volume) {
        (void) update(close, volume);
    }

    inline double last() const {
        return last_;
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &close, const Array1 &volume) {
        const std::size_t size = close.shape(0);
        require_same_size(size, volume.shape(0));
        std::vector<double> output(size);
        const auto *close_values = close.data();
        const auto *volume_values = volume.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(close_values[i]), static_cast<double>(volume_values[i]));
        }
        return make_array(std::move(output));
    }

private:
    RollingSumWindow ratios_;
    double scale_;
    double previous_close_;
    bool has_previous_;
    bool fillna_;
    double last_;
};

class SpreadFeatures {
public:
    SpreadFeatures(int realized_horizon = 5, bool fillna = true)
        : realized_horizon_(checked_horizon(realized_horizon)),
          pending_trade_price_(std::max(realized_horizon_, 1)),
          pending_side_(std::max(realized_horizon_, 1)),
          previous_trade_price_(0.0),
          has_previous_trade_(false),
          fillna_(fillna),
          last_{nan(), nan(), fillna ? 0.0 : nan()} {}

    inline SpreadFeaturesResult update(double trade_price, double bid_price, double ask_price) {
        const double midpoint = 0.5 * (bid_price + ask_price);
        const double quoted_spread = ask_price - bid_price;
        const double side = classify_side(trade_price, midpoint);
        const double effective_spread = 2.0 * std::abs(trade_price - midpoint);
        const double realized_spread = realized_spread_for_current_midpoint(midpoint, trade_price, side);

        if (realized_horizon_ > 0) {
            pending_trade_price_.push(trade_price);
            pending_side_.push(side);
        }
        previous_trade_price_ = trade_price;
        has_previous_trade_ = true;
        last_ = SpreadFeaturesResult{quoted_spread, effective_spread, realized_spread};
        return last_;
    }

    inline void advance(double trade_price, double bid_price, double ask_price) {
        (void) update(trade_price, bid_price, ask_price);
    }

    inline const SpreadFeaturesResult &last() const {
        return last_;
    }

    template <typename Array0, typename Array1, typename Array2>
    SpreadFeaturesBatchResult batch_array(const Array0 &trade_price, const Array1 &bid_price, const Array2 &ask_price) {
        const std::size_t size = trade_price.shape(0);
        require_same_size(size, bid_price.shape(0));
        require_same_size(size, ask_price.shape(0));
        std::vector<double> quoted_spread(size);
        std::vector<double> effective_spread(size);
        std::vector<double> realized_spread(size);
        const auto *trade_price_values = trade_price.data();
        const auto *bid_price_values = bid_price.data();
        const auto *ask_price_values = ask_price.data();

        for (std::size_t i = 0; i < size; ++i) {
            const SpreadFeaturesResult out = update(
                static_cast<double>(trade_price_values[i]),
                static_cast<double>(bid_price_values[i]),
                static_cast<double>(ask_price_values[i]));
            quoted_spread[i] = out.quoted_spread;
            effective_spread[i] = out.effective_spread;
            realized_spread[i] = out.realized_spread;
        }

        return {
            make_array(std::move(quoted_spread)),
            make_array(std::move(effective_spread)),
            make_array(std::move(realized_spread)),
        };
    }

private:
    static int checked_horizon(int realized_horizon) {
        if (realized_horizon < 0) {
            throw nb::value_error("realized_horizon must be non-negative");
        }
        return realized_horizon;
    }

    inline double classify_side(double trade_price, double midpoint) const {
        if (trade_price > midpoint) {
            return 1.0;
        }
        if (trade_price < midpoint) {
            return -1.0;
        }
        if (has_previous_trade_) {
            if (trade_price > previous_trade_price_) {
                return 1.0;
            }
            if (trade_price < previous_trade_price_) {
                return -1.0;
            }
        }
        return 0.0;
    }

    inline double realized_spread_for_current_midpoint(double midpoint, double trade_price, double side) const {
        if (realized_horizon_ == 0) {
            return 2.0 * side * (trade_price - midpoint);
        }
        if (!pending_trade_price_.full()) {
            return fillna_ ? 0.0 : nan();
        }
        return 2.0 * pending_side_.oldest() * (pending_trade_price_.oldest() - midpoint);
    }

    int realized_horizon_;
    RollingBuffer pending_trade_price_;
    RollingBuffer pending_side_;
    double previous_trade_price_;
    bool has_previous_trade_;
    bool fillna_;
    SpreadFeaturesResult last_;
};

class ChaikinMoneyFlow {
public:
    ChaikinMoneyFlow(int window = 20, bool fillna = true)
        : money_flow_volume_(window),
          volume_(window),
          fillna_(fillna) {}

    double update(double close, double high, double low, double volume) {
        money_flow_volume_.push(money_flow_multiplier(close, high, low) * volume);
        volume_.push(volume);
        if (!fillna_ && !money_flow_volume_.full()) {
            return nan();
        }
        return safe_divide(money_flow_volume_.sum(), volume_.sum());
    }

    nb::object batch(const InputArray &close, const InputArray &high, const InputArray &low, const InputArray &volume) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(close(i), high(i), low(i), volume(i));
        }
        return make_array(std::move(output));
    }

private:
    RollingWindow money_flow_volume_;
    RollingWindow volume_;
    bool fillna_;
};

class ForceIndex {
public:
    ForceIndex(int window = 13, bool fillna = true)
        : ema_(window, fillna),
          previous_close_(0.0),
          first_(true) {}

    double update(double close, double volume) {
        const double value = first_ ? 0.0 : (close - previous_close_) * volume;
        previous_close_ = close;
        first_ = false;
        return ema_.update(value);
    }

    nb::object batch(const InputArray &close, const InputArray &volume) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(close(i), volume(i));
        }
        return make_array(std::move(output));
    }

private:
    EMA ema_;
    double previous_close_;
    bool first_;
};

class EaseOfMovement {
public:
    EaseOfMovement(int window = 14, bool fillna = true)
        : eom_(window),
          previous_high_(0.0),
          previous_low_(0.0),
          first_(true),
          fillna_(fillna),
          last_{nan(), nan()} {}

    EaseOfMovementResult update(double high, double low, double volume) {
        update_core(high, low, volume);
        return last_;
    }

    void advance(double high, double low, double volume) {
        update_core(high, low, volume);
    }

    inline const EaseOfMovementResult &last() const {
        return last_;
    }

private:
    inline void update_core(double high, double low, double volume) {
        const double value = first_
            ? 0.0
            : safe_divide((high - previous_high_ + low - previous_low_) * (high - low), 2.0 * volume) * 100000000.0;
        previous_high_ = high;
        previous_low_ = low;
        first_ = false;
        eom_.push(value);

        last_ = {
            (!fillna_ && eom_.size() == 1) ? nan() : value,
            (!fillna_ && !eom_.full()) ? nan() : eom_.sum() / eom_.size(),
        };
    }

public:
    EaseOfMovementBatchResult batch(const InputArray &high, const InputArray &low, const InputArray &volume) {
        const std::size_t size = high.shape(0);
        std::vector<double> eom(size);
        std::vector<double> sma(size);
        for (std::size_t i = 0; i < size; ++i) {
            const EaseOfMovementResult out = update(high(i), low(i), volume(i));
            eom[i] = out.ease_of_movement;
            sma[i] = out.sma;
        }

        return {make_array(std::move(eom)), make_array(std::move(sma))};
    }

private:
    RollingWindow eom_;
    double previous_high_;
    double previous_low_;
    bool first_;
    bool fillna_;
    EaseOfMovementResult last_;
};

class VolumePriceTrend {
public:
    VolumePriceTrend(int smoothing_window = 0, bool fillna = true)
        : smoothing_(std::max(smoothing_window, 1)),
          smoothing_window_(smoothing_window),
          value_(0.0),
          previous_close_(0.0),
          first_(true),
          fillna_(fillna) {}

    double update(double close, double volume) {
        if (!first_) {
            value_ += safe_divide(close - previous_close_, previous_close_) * volume;
        }
        previous_close_ = close;
        first_ = false;

        if (smoothing_window_ <= 0) {
            return value_;
        }

        smoothing_.push(value_);
        if (!fillna_ && !smoothing_.full()) {
            return nan();
        }
        return smoothing_.sum() / smoothing_.size();
    }

    nb::object batch(const InputArray &close, const InputArray &volume) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(close(i), volume(i));
        }
        return make_array(std::move(output));
    }

private:
    RollingWindow smoothing_;
    int smoothing_window_;
    double value_;
    double previous_close_;
    bool first_;
    bool fillna_;
};

class NegativeVolumeIndex {
public:
    NegativeVolumeIndex()
        : value_(1000.0),
          previous_close_(0.0),
          previous_volume_(0.0),
          first_(true) {}

    double update(double close, double volume) {
        if (!first_ && volume < previous_volume_) {
            value_ *= 1.0 + safe_divide(close - previous_close_, previous_close_);
        }

        previous_close_ = close;
        previous_volume_ = volume;
        first_ = false;
        return value_;
    }

    nb::object batch(const InputArray &close, const InputArray &volume) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(close(i), volume(i));
        }
        return make_array(std::move(output));
    }

private:
    double value_;
    double previous_close_;
    double previous_volume_;
    bool first_;
};

class VolumeWeightedAveragePrice {
public:
    VolumeWeightedAveragePrice(int window = 14, bool fillna = true)
        : typical_price_volume_(window),
          volume_(window),
          fillna_(fillna) {}

    double update(double close, double high, double low, double volume) {
        const double typical_price = (high + low + close) / 3.0;
        typical_price_volume_.push(typical_price * volume);
        volume_.push(volume);

        if (!fillna_ && !typical_price_volume_.full()) {
            return nan();
        }
        return safe_divide(typical_price_volume_.sum(), volume_.sum());
    }

    nb::object batch(const InputArray &close, const InputArray &high, const InputArray &low, const InputArray &volume) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(close(i), high(i), low(i), volume(i));
        }
        return make_array(std::move(output));
    }

private:
    RollingWindow typical_price_volume_;
    RollingWindow volume_;
    bool fillna_;
};

class AnchoredVWAP {
public:
    AnchoredVWAP()
        : cumulative_price_volume_(0.0),
          cumulative_volume_(0.0),
          last_(nan()) {}

    double update(double close, double high, double low, double volume, double anchor) {
        if (anchor != 0.0 || cumulative_volume_ == 0.0) {
            cumulative_price_volume_ = 0.0;
            cumulative_volume_ = 0.0;
        }

        const double typical_price = (high + low + close) / 3.0;
        cumulative_price_volume_ += typical_price * volume;
        cumulative_volume_ += volume;
        last_ = safe_divide(cumulative_price_volume_, cumulative_volume_);
        return last_;
    }

    void advance(double close, double high, double low, double volume, double anchor) {
        (void) update(close, high, low, volume, anchor);
    }

    double last() const {
        return last_;
    }

    template <typename Array0, typename Array1, typename Array2, typename Array3, typename Array4>
    nb::object batch_array(
        const Array0 &close,
        const Array1 &high,
        const Array2 &low,
        const Array3 &volume,
        const Array4 &anchor) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        require_same_size(size, volume.shape(0));
        require_same_size(size, anchor.shape(0));
        std::vector<double> output(size);
        const auto *close_values = close.data();
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        const auto *volume_values = volume.data();
        const auto *anchor_values = anchor.data();

        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(
                static_cast<double>(close_values[i]),
                static_cast<double>(high_values[i]),
                static_cast<double>(low_values[i]),
                static_cast<double>(volume_values[i]),
                static_cast<double>(anchor_values[i]));
        }

        return make_array(std::move(output));
    }

    template <typename Array0, typename Array1, typename Array2, typename Array3, typename Array4>
    double replay_update_array(
        const Array0 &close,
        const Array1 &high,
        const Array2 &low,
        const Array3 &volume,
        const Array4 &anchor) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        require_same_size(size, volume.shape(0));
        require_same_size(size, anchor.shape(0));
        const auto *close_values = close.data();
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        const auto *volume_values = volume.data();
        const auto *anchor_values = anchor.data();
        double checksum = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            checksum += update(
                static_cast<double>(close_values[i]),
                static_cast<double>(high_values[i]),
                static_cast<double>(low_values[i]),
                static_cast<double>(volume_values[i]),
                static_cast<double>(anchor_values[i]));
        }
        return checksum;
    }

    template <typename Array0, typename Array1, typename Array2, typename Array3, typename Array4>
    double replay_advance_array(
        const Array0 &close,
        const Array1 &high,
        const Array2 &low,
        const Array3 &volume,
        const Array4 &anchor) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        require_same_size(size, volume.shape(0));
        require_same_size(size, anchor.shape(0));
        const auto *close_values = close.data();
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        const auto *volume_values = volume.data();
        const auto *anchor_values = anchor.data();
        double checksum = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            advance(
                static_cast<double>(close_values[i]),
                static_cast<double>(high_values[i]),
                static_cast<double>(low_values[i]),
                static_cast<double>(volume_values[i]),
                static_cast<double>(anchor_values[i]));
            checksum += result_checksum(last_);
        }
        return checksum;
    }

private:
    double cumulative_price_volume_;
    double cumulative_volume_;
    double last_;
};

class VolumeProfile {
public:
    VolumeProfile(int window = 128, int bins = 24, double value_area_percent = 70.0, bool fillna = true)
        : window_(checked_window(window)),
          bins_(checked_bins(bins)),
          value_area_fraction_(checked_value_area_percent(value_area_percent) / 100.0),
          fillna_(fillna),
          prices_(window_),
          volumes_(window_),
          histogram_(bins_),
          count_(0),
          next_(0),
          last_{nan(), nan(), nan()} {}

    VolumeProfileResult update(double close, double volume) {
        prices_[next_] = close;
        volumes_[next_] = volume;
        if (count_ < window_) {
            ++count_;
        }
        next_ = (next_ + 1) % window_;

        if (!fillna_ && count_ < window_) {
            last_ = VolumeProfileResult{nan(), nan(), nan()};
            return last_;
        }

        last_ = compute();
        return last_;
    }

    void advance(double close, double volume) {
        (void) update(close, volume);
    }

    const VolumeProfileResult &last() const {
        return last_;
    }

    template <typename Array0, typename Array1>
    VolumeProfileBatchResult batch_array(const Array0 &close, const Array1 &volume) {
        const std::size_t size = close.shape(0);
        require_same_size(size, volume.shape(0));
        std::vector<double> point_of_control(size);
        std::vector<double> value_area_high(size);
        std::vector<double> value_area_low(size);
        const auto *close_values = close.data();
        const auto *volume_values = volume.data();

        for (std::size_t i = 0; i < size; ++i) {
            const VolumeProfileResult out = update(
                static_cast<double>(close_values[i]),
                static_cast<double>(volume_values[i]));
            point_of_control[i] = out.point_of_control;
            value_area_high[i] = out.value_area_high;
            value_area_low[i] = out.value_area_low;
        }

        return {
            make_array(std::move(point_of_control)),
            make_array(std::move(value_area_high)),
            make_array(std::move(value_area_low)),
        };
    }

private:
    static std::size_t checked_window(int window) {
        if (window <= 0) {
            throw nb::value_error("window must be positive");
        }
        return static_cast<std::size_t>(window);
    }

    static std::size_t checked_bins(int bins) {
        if (bins <= 0) {
            throw nb::value_error("bins must be positive");
        }
        return static_cast<std::size_t>(bins);
    }

    static double checked_value_area_percent(double value_area_percent) {
        if (value_area_percent <= 0.0 || value_area_percent > 100.0) {
            throw nb::value_error("value_area_percent must be in the range (0, 100]");
        }
        return value_area_percent;
    }

    VolumeProfileResult compute() {
        double min_price = prices_[0];
        double max_price = prices_[0];
        double total_volume = 0.0;
        for (std::size_t i = 0; i < count_; ++i) {
            min_price = std::min(min_price, prices_[i]);
            max_price = std::max(max_price, prices_[i]);
            total_volume += volumes_[i];
        }

        if (total_volume == 0.0 || min_price == max_price) {
            return VolumeProfileResult{min_price, max_price, min_price};
        }

        std::fill(histogram_.begin(), histogram_.end(), 0.0);
        const double width = (max_price - min_price) / static_cast<double>(bins_);
        for (std::size_t i = 0; i < count_; ++i) {
            std::size_t bin = static_cast<std::size_t>((prices_[i] - min_price) / width);
            if (bin >= bins_) {
                bin = bins_ - 1;
            }
            histogram_[bin] += volumes_[i];
        }

        std::size_t point_of_control_bin = 0;
        for (std::size_t i = 1; i < bins_; ++i) {
            if (histogram_[i] > histogram_[point_of_control_bin]) {
                point_of_control_bin = i;
            }
        }

        std::size_t low_bin = point_of_control_bin;
        std::size_t high_bin = point_of_control_bin;
        double cumulative_volume = histogram_[point_of_control_bin];
        const double target_volume = total_volume * value_area_fraction_;
        while (cumulative_volume < target_volume && (low_bin > 0 || high_bin + 1 < bins_)) {
            const double left_volume = low_bin > 0 ? histogram_[low_bin - 1] : -1.0;
            const double right_volume = high_bin + 1 < bins_ ? histogram_[high_bin + 1] : -1.0;
            if (right_volume > left_volume) {
                ++high_bin;
                cumulative_volume += right_volume;
            } else {
                --low_bin;
                cumulative_volume += left_volume;
            }
        }

        const double point_of_control = min_price + (static_cast<double>(point_of_control_bin) + 0.5) * width;
        const double value_area_low = min_price + static_cast<double>(low_bin) * width;
        const double value_area_high = min_price + static_cast<double>(high_bin + 1) * width;
        return VolumeProfileResult{point_of_control, value_area_high, value_area_low};
    }

    std::size_t window_;
    std::size_t bins_;
    double value_area_fraction_;
    bool fillna_;
    std::vector<double> prices_;
    std::vector<double> volumes_;
    std::vector<double> histogram_;
    std::size_t count_;
    std::size_t next_;
    VolumeProfileResult last_;
};

class VolumeWeightedMovingAverage {
public:
    VolumeWeightedMovingAverage(int window = 20, bool fillna = true)
        : price_volume_(window),
          volume_(window),
          fillna_(fillna) {}

    double update(double close, double volume) {
        price_volume_.push(close * volume);
        volume_.push(volume);

        if (!fillna_ && !price_volume_.full()) {
            return nan();
        }
        return safe_divide(price_volume_.sum(), volume_.sum());
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &close, const Array1 &volume) {
        const std::size_t size = close.shape(0);
        require_same_size(size, volume.shape(0));
        std::vector<double> output(size);
        const auto *close_values = close.data();
        const auto *volume_values = volume.data();

        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(close_values[i]), static_cast<double>(volume_values[i]));
        }

        return make_array(std::move(output));
    }

private:
    RollingSumWindow price_volume_;
    RollingSumWindow volume_;
    bool fillna_;
};

class Momentum {
public:
    Momentum(int window = 10, bool fillna = true)
        : close_(window),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double close) {
        const double previous = close_.full() ? close_.oldest() : nan();
        close_.push(close);
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan || std::isnan(previous) ? nan() : close - previous;
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        const auto *values = close.data();
        const long start_counter = counter_;
        if (start_counter == 0 && close_.size() == 0 && window_ > 0) {
            const std::size_t window = static_cast<std::size_t>(window_);
            for (std::size_t i = 0; i < size; ++i) {
                output[i] = i < window ? nan() : static_cast<double>(values[i]) - static_cast<double>(values[i - window]);
            }

            const std::size_t keep = std::min<std::size_t>(window, size);
            close_.reset();
            for (std::size_t i = size - keep; i < size; ++i) {
                close_.push(static_cast<double>(values[i]));
            }
            counter_ = static_cast<long>(size);
            return make_array(std::move(output));
        }

        batch_kernels::momentum(values, size, close_, window_, fillna_, start_counter, output.data());

        const std::size_t keep = std::min<std::size_t>(static_cast<std::size_t>(std::max(window_, 0)), size);
        for (std::size_t i = size - keep; i < size; ++i) {
            close_.push(static_cast<double>(values[i]));
        }

        counter_ = start_counter + static_cast<long>(size);
        return make_array(std::move(output));
    }

private:
    RollingBuffer close_;
    int window_;
    bool fillna_;
    long counter_;
};

class RateOfChangePercentage {
public:
    RateOfChangePercentage(int window = 10, bool fillna = true)
        : close_(window, true),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double close) {
        const double previous = close_.update(close);
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : safe_divide(close - previous, previous);
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        batch_kernels::rate_of_change(close.data(), size, close_, window_, fillna_, counter_, 1.0, true, output.data());
        return make_array(std::move(output));
    }

private:
    Delay close_;
    int window_;
    bool fillna_;
    long counter_;
};

class RateOfChangeRatio {
public:
    RateOfChangeRatio(int window = 10, bool fillna = true)
        : close_(window, true),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double close) {
        const double previous = close_.update(close);
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : safe_divide(close, previous);
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        return batch_scaled(close, 1.0);
    }

    template <typename Array>
    nb::object batch_scaled(const Array &close, double scale) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        batch_kernels::rate_of_change(close.data(), size, close_, window_, fillna_, counter_, scale, false, output.data());
        return make_array(std::move(output));
    }

private:
    Delay close_;
    int window_;
    bool fillna_;
    long counter_;
};

class RateOfChangeRatio100 {
public:
    RateOfChangeRatio100(int window = 10, bool fillna = true)
        : rocr_(window, fillna) {}

    double update(double close) {
        return 100.0 * rocr_.update(close);
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        return rocr_.batch_scaled(close, 100.0);
    }

private:
    RateOfChangeRatio rocr_;
};

class AbsolutePriceOscillator {
public:
    AbsolutePriceOscillator(double fast = 12.0, double slow = 26.0)
        : fast_(fast, true),
          slow_(slow, true) {}

    double update(double close) {
        return fast_.update(close) - slow_.update(close);
    }

private:
    EMA fast_;
    EMA slow_;
};

class MACDFix {
public:
    explicit MACDFix(int signal = 9, bool fillna = false)
        : macd_(12, 26, signal, fillna) {}

    double update(double close) {
        return macd_.update(close);
    }

private:
    MACD macd_;
};

class DoubleEMA {
public:
    DoubleEMA(double window = 30.0, bool fillna = true)
        : ema1_(window, true),
          ema2_(window, true),
          window_(static_cast<int>(window)),
          fillna_(fillna),
          counter_(0) {}

    double update(double value) {
        const double ema1 = ema1_.update(value);
        const double ema2 = ema2_.update(ema1);
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : 2.0 * ema1 - ema2;
    }

private:
    EMA ema1_;
    EMA ema2_;
    int window_;
    bool fillna_;
    long counter_;
};

class TripleEMA {
public:
    TripleEMA(double window = 30.0, bool fillna = true)
        : ema1_(window, true),
          ema2_(window, true),
          ema3_(window, true),
          window_(static_cast<int>(window)),
          fillna_(fillna),
          counter_(0) {}

    double update(double value) {
        const double ema1 = ema1_.update(value);
        const double ema2 = ema2_.update(ema1);
        const double ema3 = ema3_.update(ema2);
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : 3.0 * ema1 - 3.0 * ema2 + ema3;
    }

private:
    EMA ema1_;
    EMA ema2_;
    EMA ema3_;
    int window_;
    bool fillna_;
    long counter_;
};

class T3MovingAverage {
public:
    T3MovingAverage(double window = 5.0, double vfactor = 0.7, bool fillna = true)
        : e1_(window, true),
          e2_(window, true),
          e3_(window, true),
          e4_(window, true),
          e5_(window, true),
          e6_(window, true),
          vfactor_(vfactor),
          window_(static_cast<int>(window)),
          fillna_(fillna),
          counter_(0) {}

    double update(double value) {
        const double e1 = e1_.update(value);
        const double e2 = e2_.update(e1);
        const double e3 = e3_.update(e2);
        const double e4 = e4_.update(e3);
        const double e5 = e5_.update(e4);
        const double e6 = e6_.update(e5);
        const double v2 = vfactor_ * vfactor_;
        const double v3 = v2 * vfactor_;
        const double c1 = -v3;
        const double c2 = 3.0 * v2 + 3.0 * v3;
        const double c3 = -6.0 * v2 - 3.0 * vfactor_ - 3.0 * v3;
        const double c4 = 1.0 + 3.0 * vfactor_ + v3 + 3.0 * v2;
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
    }

    template <typename Array>
    nb::object batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        batch_kernels::t3_moving_average(
            input.data(),
            size,
            e1_,
            e2_,
            e3_,
            e4_,
            e5_,
            e6_,
            vfactor_,
            window_,
            fillna_,
            counter_,
            output.data());
        return make_array(std::move(output));
    }

private:
    EMA e1_;
    EMA e2_;
    EMA e3_;
    EMA e4_;
    EMA e5_;
    EMA e6_;
    double vfactor_;
    int window_;
    bool fillna_;
    long counter_;
};

class Trix {
public:
    Trix(double window = 30.0, bool fillna = true)
        : ema1_(window, true),
          ema2_(window, true),
          ema3_(window, true),
          previous_(1, false),
          window_(static_cast<int>(window)),
          fillna_(fillna),
          counter_(0) {}

    double update(double value) {
        const double triple = ema3_.update(ema2_.update(ema1_.update(value)));
        const double previous = previous_.update(triple);
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        if (return_nan || std::isnan(previous) || previous == 0.0) {
            return return_nan ? nan() : 0.0;
        }
        return 100.0 * (triple - previous) / previous;
    }

private:
    EMA ema1_;
    EMA ema2_;
    EMA ema3_;
    Delay previous_;
    int window_;
    bool fillna_;
    long counter_;
};

class SchaffTrendCycle {
public:
    SchaffTrendCycle(
        int slow = 50,
        int fast = 23,
        int cycle = 10,
        int smooth1 = 3,
        int smooth2 = 3,
        bool fillna = true)
        : fast_(fast, true),
          slow_(slow, true),
          macd_(cycle),
          stoch_d_window_(cycle),
          stoch_d_(smooth1, fillna),
          stc_(smooth2, fillna),
          ready_(slow + 2 * cycle + smooth1 + smooth2),
          fillna_(fillna),
          counter_(0) {}

    double update(double close) {
        const double macd = fast_.update(close) - slow_.update(close);
        macd_.push(macd);
        const double macd_min = macd_.min();
        const double macd_max = macd_.max();
        const double stoch_k = safe_divide(100.0 * (macd - macd_min), macd_max - macd_min);
        const double stoch_d = stoch_d_.update(stoch_k);
        stoch_d_window_.push(stoch_d);
        const double stoch_d_min = stoch_d_window_.min();
        const double stoch_d_max = stoch_d_window_.max();
        const double stoch_kd = safe_divide(100.0 * (stoch_d - stoch_d_min), stoch_d_max - stoch_d_min);
        const double retval = stc_.update(stoch_kd);
        const bool return_nan = !fillna_ && counter_ < ready_;
        ++counter_;
        return return_nan ? nan() : retval;
    }

    nb::object batch(const InputArray &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(close(i));
        }
        return make_array(std::move(output));
    }

private:
    EMA fast_;
    EMA slow_;
    RollingWindow macd_;
    RollingWindow stoch_d_window_;
    EMA stoch_d_;
    EMA stc_;
    int ready_;
    bool fillna_;
    long counter_;
};

class WeightedMovingAverage {
public:
    WeightedMovingAverage(int window = 30, bool fillna = true)
        : window_(window),
          values_(window),
          fillna_(fillna),
          weighted_sum_(0.0),
          sum_(0.0) {}

    double update(double value) {
        const std::size_t size = values_.size();
        if (values_.full()) {
            const double oldest = values_.oldest();
            weighted_sum_ = weighted_sum_ - sum_ + static_cast<double>(window_) * value;
            sum_ += value - oldest;
        } else {
            weighted_sum_ += static_cast<double>(size + 1) * value;
            sum_ += value;
        }

        values_.push(value);
        if (!fillna_ && !values_.full()) {
            return nan();
        }

        const double n = static_cast<double>(values_.size());
        return weighted_sum_ / (n * (n + 1.0) * 0.5);
    }

    template <typename Array>
    nb::object batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        batch_kernels::weighted_moving_average(
            input.data(),
            size,
            window_,
            values_,
            fillna_,
            weighted_sum_,
            sum_,
            output.data());
        return make_array(std::move(output));
    }

private:
    int window_;
    RollingBuffer values_;
    bool fillna_;
    double weighted_sum_;
    double sum_;
};

class HullMovingAverage {
public:
    HullMovingAverage(int window = 30, bool fillna = true)
        : half_(std::max(window / 2, 1), true),
          full_(std::max(window, 1), true),
          hull_(std::max(static_cast<int>(std::sqrt(static_cast<double>(std::max(window, 1)))), 1), true),
          warmup_(std::max(window, 1) + std::max(static_cast<int>(std::sqrt(static_cast<double>(std::max(window, 1)))), 1) - 1),
          fillna_(fillna),
          count_(0) {}

    double update(double value) {
        const double transformed = 2.0 * half_.update(value) - full_.update(value);
        const double output = hull_.update(transformed);
        ++count_;
        return (!fillna_ && count_ < warmup_) ? nan() : output;
    }

    template <typename Array>
    nb::object batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        const auto *values = input.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(values[i]));
        }
        return make_array(std::move(output));
    }

private:
    WeightedMovingAverage half_;
    WeightedMovingAverage full_;
    WeightedMovingAverage hull_;
    int warmup_;
    bool fillna_;
    long count_;
};

class TriangularMovingAverage {
public:
    TriangularMovingAverage(int window = 30, bool fillna = true)
        : first_(window % 2 == 0 ? window / 2 : (window + 1) / 2, fillna),
          second_(window % 2 == 0 ? window / 2 + 1 : (window + 1) / 2, fillna) {}

    double update(double value) {
        return second_.update(first_.update(value));
    }

private:
    SMA first_;
    SMA second_;
};

class ChandeMomentumOscillator {
public:
    ChandeMomentumOscillator(int window = 14, bool fillna = true)
        : gains_(window),
          losses_(window),
          window_(window),
          gain_sum_(0.0),
          loss_sum_(0.0),
          previous_(0.0),
          first_(true),
          fillna_(fillna) {}

    double update(double close) {
        double gain = 0.0;
        double loss = 0.0;
        if (!first_) {
            const double change = close - previous_;
            if (change > 0.0) {
                gain = change;
            } else {
                loss = -change;
            }
        }

        rolling_sum_push(gains_, gain_sum_, gain);
        rolling_sum_push(losses_, loss_sum_, loss);
        previous_ = close;
        first_ = false;

        if (!fillna_ && !gains_.full()) {
            return nan();
        }

        return safe_divide(100.0 * (gain_sum_ - loss_sum_), gain_sum_ + loss_sum_);
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        if (first_ && gains_.size() == 0) {
            batch_kernels::chande_momentum_oscillator_fresh(
                close.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                fillna_,
                output.data());
            rebuild_state(close.data(), size);
        } else {
            batch_kernels::chande_momentum_oscillator(
                close.data(),
                size,
                gains_,
                losses_,
                gain_sum_,
                loss_sum_,
                previous_,
                first_,
                fillna_,
                output.data());
        }
        return make_array(std::move(output));
    }

private:
    template <typename Value>
    void rebuild_state(const Value *values, std::size_t size) {
        if (size == 0) {
            return;
        }

        gains_.reset();
        losses_.reset();
        gain_sum_ = 0.0;
        loss_sum_ = 0.0;
        const std::size_t window = static_cast<std::size_t>(std::max(window_, 1));
        const std::size_t start = size > window ? size - window : 0;
        for (std::size_t i = start; i < size; ++i) {
            double gain = 0.0;
            double loss = 0.0;
            if (i > 0) {
                const double change = static_cast<double>(values[i]) - static_cast<double>(values[i - 1]);
                if (change > 0.0) {
                    gain = change;
                } else {
                    loss = -change;
                }
            }
            gains_.push(gain);
            losses_.push(loss);
            gain_sum_ += gain;
            loss_sum_ += loss;
        }
        previous_ = static_cast<double>(values[size - 1]);
        first_ = false;
    }

    RollingBuffer gains_;
    RollingBuffer losses_;
    int window_;
    double gain_sum_;
    double loss_sum_;
    double previous_;
    bool first_;
    bool fillna_;
};

class CommodityChannelIndex {
public:
    CommodityChannelIndex(int window = 14, bool fillna = true)
        : typical_(window),
          fillna_(fillna) {}

    double update(double close, double high, double low) {
        const double typical = (high + low + close) / 3.0;
        typical_.push(typical);
        if (!fillna_ && !typical_.full()) {
            return nan();
        }

        const double mean = typical_.sum() / typical_.size();
        double mean_deviation = 0.0;
        for (std::size_t i = 0; i < typical_.size(); ++i) {
            mean_deviation += std::abs(typical_.at(i) - mean);
        }
        mean_deviation /= typical_.size();
        return safe_divide(typical - mean, 0.015 * mean_deviation);
    }

private:
    RollingWindow typical_;
    bool fillna_;
};

class MoneyFlowIndex {
public:
    MoneyFlowIndex(int window = 14, bool fillna = true)
        : positive_(window),
          negative_(window),
          window_(window),
          positive_sum_(0.0),
          negative_sum_(0.0),
          previous_typical_(0.0),
          first_(true),
          fillna_(fillna) {}

    double update(double close, double high, double low, double volume) {
        const double typical = (high + low + close) / 3.0;
        const double money_flow = typical * volume;
        double positive = 0.0;
        double negative = 0.0;

        if (!first_) {
            if (typical > previous_typical_) {
                positive = money_flow;
            } else if (typical < previous_typical_) {
                negative = money_flow;
            }
        }

        rolling_sum_push(positive_, positive_sum_, positive);
        rolling_sum_push(negative_, negative_sum_, negative);
        previous_typical_ = typical;
        first_ = false;

        if (!fillna_ && !positive_.full()) {
            return nan();
        }

        if (negative_sum_ == 0.0) {
            return positive_sum_ == 0.0 ? 50.0 : 100.0;
        }
        return 100.0 - 100.0 / (1.0 + positive_sum_ / negative_sum_);
    }

    template <typename Array0, typename Array1, typename Array2, typename Array3>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low, const Array3 &volume) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        require_same_size(size, volume.shape(0));
        std::vector<double> output(size);
        if (first_ && positive_.size() == 0) {
            batch_kernels::money_flow_index_fresh(
                close.data(),
                high.data(),
                low.data(),
                volume.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                fillna_,
                output.data());
            rebuild_state(close.data(), high.data(), low.data(), volume.data(), size);
        } else {
            batch_kernels::money_flow_index(
                close.data(),
                high.data(),
                low.data(),
                volume.data(),
                size,
                positive_,
                negative_,
                positive_sum_,
                negative_sum_,
                previous_typical_,
                first_,
                fillna_,
                output.data());
        }
        return make_array(std::move(output));
    }

private:
    template <typename Close, typename High, typename Low, typename Volume>
    void rebuild_state(const Close *close, const High *high, const Low *low, const Volume *volume, std::size_t size) {
        if (size == 0) {
            return;
        }

        positive_.reset();
        negative_.reset();
        positive_sum_ = 0.0;
        negative_sum_ = 0.0;
        const std::size_t window = static_cast<std::size_t>(std::max(window_, 1));
        const std::size_t start = size > window ? size - window : 0;
        for (std::size_t i = start; i < size; ++i) {
            const double typical = (static_cast<double>(high[i]) + static_cast<double>(low[i]) + static_cast<double>(close[i])) / 3.0;
            double positive = 0.0;
            double negative = 0.0;
            if (i > 0) {
                const double previous_typical = (
                    static_cast<double>(high[i - 1]) +
                    static_cast<double>(low[i - 1]) +
                    static_cast<double>(close[i - 1])) / 3.0;
                const double money_flow = typical * static_cast<double>(volume[i]);
                if (typical > previous_typical) {
                    positive = money_flow;
                } else if (typical < previous_typical) {
                    negative = money_flow;
                }
            }
            positive_.push(positive);
            negative_.push(negative);
            positive_sum_ += positive;
            negative_sum_ += negative;
        }
        previous_typical_ = (
            static_cast<double>(high[size - 1]) +
            static_cast<double>(low[size - 1]) +
            static_cast<double>(close[size - 1])) / 3.0;
        first_ = false;
    }

    RollingBuffer positive_;
    RollingBuffer negative_;
    int window_;
    double positive_sum_;
    double negative_sum_;
    double previous_typical_;
    bool first_;
    bool fillna_;
};

class RollingPairStats {
public:
    explicit RollingPairStats(int window)
        : x_(window),
          y_(window),
          sum_x_(0.0),
          sum_y_(0.0),
          sum_x2_(0.0),
          sum_y2_(0.0),
          sum_xy_(0.0) {}

    void push(double x, double y) {
        if (x_.full()) {
            const double old_x = x_.oldest();
            const double old_y = y_.oldest();
            sum_x_ -= old_x;
            sum_y_ -= old_y;
            sum_x2_ -= old_x * old_x;
            sum_y2_ -= old_y * old_y;
            sum_xy_ -= old_x * old_y;
        }

        x_.push(x);
        y_.push(y);
        sum_x_ += x;
        sum_y_ += y;
        sum_x2_ += x * x;
        sum_y2_ += y * y;
        sum_xy_ += x * y;
    }

    bool full() const {
        return x_.full();
    }

    std::size_t size() const {
        return x_.size();
    }

    double correlation() const {
        const std::size_t n = size();
        const double n_value = static_cast<double>(n);
        const double numerator = n_value * sum_xy_ - sum_x_ * sum_y_;
        const double denominator = std::sqrt((n_value * sum_x2_ - sum_x_ * sum_x_) * (n_value * sum_y2_ - sum_y_ * sum_y_));
        return safe_divide(numerator, denominator);
    }

    double beta() const {
        const std::size_t n = size();
        const double n_value = static_cast<double>(n);
        const double covariance = n_value * sum_xy_ - sum_x_ * sum_y_;
        const double variance_y = n_value * sum_y2_ - sum_y_ * sum_y_;
        return safe_divide(covariance, variance_y);
    }

    std::size_t capacity() const {
        return x_.capacity();
    }

    void reset() {
        x_.reset();
        y_.reset();
        sum_x_ = 0.0;
        sum_y_ = 0.0;
        sum_x2_ = 0.0;
        sum_y2_ = 0.0;
        sum_xy_ = 0.0;
    }

private:
    RollingBuffer x_;
    RollingBuffer y_;
    double sum_x_;
    double sum_y_;
    double sum_x2_;
    double sum_y2_;
    double sum_xy_;
};

class Correlation {
public:
    Correlation(int window = 30, bool fillna = true)
        : stats_(window),
          fillna_(fillna) {}

    double update(double real0, double real1) {
        stats_.push(real0, real1);
        if (!fillna_ && !stats_.full()) {
            return nan();
        }
        return stats_.correlation();
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &real0, const Array1 &real1) {
        const std::size_t size = real0.shape(0);
        require_same_size(size, real1.shape(0));
        std::vector<double> output(size);
        if (stats_.size() == 0) {
            const std::size_t window = stats_.capacity();
            batch_kernels::rolling_pair_fresh(real0.data(), real1.data(), size, window, fillna_, false, output.data());
            stats_.reset();
            const std::size_t start = size > window ? size - window : 0;
            for (std::size_t i = start; i < size; ++i) {
                stats_.push(static_cast<double>(real0.data()[i]), static_cast<double>(real1.data()[i]));
            }
        } else {
            const auto *x = real0.data();
            const auto *y = real1.data();
            for (std::size_t i = 0; i < size; ++i) {
                output[i] = update(static_cast<double>(x[i]), static_cast<double>(y[i]));
            }
        }
        return make_array(std::move(output));
    }

private:
    RollingPairStats stats_;
    bool fillna_;
};

class Beta {
public:
    Beta(int window = 5, bool fillna = true)
        : stats_(window),
          fillna_(fillna) {}

    double update(double real0, double real1) {
        stats_.push(real0, real1);
        if (!fillna_ && !stats_.full()) {
            return nan();
        }
        return stats_.beta();
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &real0, const Array1 &real1) {
        const std::size_t size = real0.shape(0);
        require_same_size(size, real1.shape(0));
        std::vector<double> output(size);
        if (stats_.size() == 0) {
            const std::size_t window = stats_.capacity();
            batch_kernels::rolling_pair_fresh(real0.data(), real1.data(), size, window, fillna_, true, output.data());
            stats_.reset();
            const std::size_t start = size > window ? size - window : 0;
            for (std::size_t i = start; i < size; ++i) {
                stats_.push(static_cast<double>(real0.data()[i]), static_cast<double>(real1.data()[i]));
            }
        } else {
            const auto *x = real0.data();
            const auto *y = real1.data();
            for (std::size_t i = 0; i < size; ++i) {
                output[i] = update(static_cast<double>(x[i]), static_cast<double>(y[i]));
            }
        }
        return make_array(std::move(output));
    }

private:
    RollingPairStats stats_;
    bool fillna_;
};

class Variance {
public:
    Variance(int window = 5, bool fillna = true)
        : values_(window),
          fillna_(fillna),
          sum_(0.0),
          sum2_(0.0) {}

    double update(double value) {
        if (values_.full()) {
            const double oldest = values_.oldest();
            sum_ -= oldest;
            sum2_ -= oldest * oldest;
        }

        values_.push(value);
        sum_ += value;
        sum2_ += value * value;

        if (!fillna_ && !values_.full()) {
            return nan();
        }

        const double n = static_cast<double>(values_.size());
        const double variance = (sum2_ - sum_ * sum_ / n) / n;
        return variance < 0.0 && variance > -1e-12 ? 0.0 : variance;
    }

    template <typename Array>
    nb::object batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        if (values_.size() == 0) {
            const std::size_t window = values_.capacity();
            batch_kernels::rolling_variance_fresh(input.data(), size, window, fillna_, output.data());
            batch_kernels::rebuild_buffer_sum(input.data(), size, window, values_, sum_, sum2_);
        } else {
            batch_kernels::rolling_variance(input.data(), size, values_, fillna_, sum_, sum2_, output.data());
        }
        return make_array(std::move(output));
    }

private:
    RollingBuffer values_;
    bool fillna_;
    double sum_;
    double sum2_;
};

class LinearRegressionCore {
public:
    LinearRegressionCore(int window = 14, bool fillna = true)
        : values_(window),
          window_(std::max(window, 1)),
          fillna_(fillna),
          sum_y_(0.0),
          sum_xy_(0.0) {}

    LinearRegressionResult update(double value) {
        if (values_.full()) {
            const double oldest = values_.oldest();
            const double old_sum_y = sum_y_;
            sum_xy_ = sum_xy_ - (old_sum_y - oldest) + static_cast<double>(window_ - 1) * value;
            sum_y_ = old_sum_y - oldest + value;
        } else {
            sum_xy_ += static_cast<double>(values_.size()) * value;
            sum_y_ += value;
        }

        values_.push(value);
        if (!fillna_ && !values_.full()) {
            return {nan(), nan(), nan(), nan(), nan()};
        }

        const double n = static_cast<double>(values_.size());
        const double sum_x = n * (n - 1.0) * 0.5;
        const double sum_x2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;

        const double denominator = n * sum_x2 - sum_x * sum_x;
        const double slope = safe_divide(n * sum_xy_ - sum_x * sum_y_, denominator);
        const double intercept = (sum_y_ - slope * sum_x) / n;
        const double value_out = intercept + slope * (n - 1.0);

        return {
            value_out,
            slope,
            intercept,
            std::atan(slope) * 180.0 / std::acos(-1.0),
            intercept + slope * n,
        };
    }

private:
    RollingBuffer values_;
    int window_;
    bool fillna_;
    double sum_y_;
    double sum_xy_;
};

class LinearRegression {
public:
    LinearRegression(int window = 14, bool fillna = true)
        : core_(window, fillna) {}

    double update(double value) {
        return core_.update(value).value;
    }

private:
    LinearRegressionCore core_;
};

class LinearRegressionSlope {
public:
    LinearRegressionSlope(int window = 14, bool fillna = true)
        : core_(window, fillna) {}

    double update(double value) {
        return core_.update(value).slope;
    }

private:
    LinearRegressionCore core_;
};

class LinearRegressionIntercept {
public:
    LinearRegressionIntercept(int window = 14, bool fillna = true)
        : core_(window, fillna) {}

    double update(double value) {
        return core_.update(value).intercept;
    }

private:
    LinearRegressionCore core_;
};

class LinearRegressionAngle {
public:
    LinearRegressionAngle(int window = 14, bool fillna = true)
        : core_(window, fillna) {}

    double update(double value) {
        return core_.update(value).angle;
    }

private:
    LinearRegressionCore core_;
};

class TimeSeriesForecast {
public:
    TimeSeriesForecast(int window = 14, bool fillna = true)
        : core_(window, fillna) {}

    double update(double value) {
        return core_.update(value).tsf;
    }

private:
    LinearRegressionCore core_;
};

class RollingMinMax {
public:
    RollingMinMax(int window = 30, bool fillna = true)
        : values_(window),
          fillna_(fillna),
          counter_(-1) {}

    RollingMinMaxResult update(double value) {
        ++counter_;
        values_.push(value);
        if (!fillna_ && !values_.full()) {
            return {nan(), nan(), nan(), nan()};
        }

        const std::size_t min_offset = values_.min_offset();
        const std::size_t max_offset = values_.max_offset();
        const long base = counter_ - static_cast<long>(values_.size()) + 1;
        return {
            values_.at(min_offset),
            values_.at(max_offset),
            static_cast<double>(base + static_cast<long>(min_offset)),
            static_cast<double>(base + static_cast<long>(max_offset)),
        };
    }

private:
    RollingWindow values_;
    bool fillna_;
    long counter_;
};

class HighIndex {
public:
    HighIndex(int window = 30, bool fillna = true)
        : values_(window, true),
          window_(window),
          fillna_(fillna),
          counter_(-1) {}

    double update(double value) {
        ++counter_;
        values_.push(value);
        if (!fillna_ && !values_.full()) {
            return nan();
        }
        const long base = counter_ - static_cast<long>(values_.size()) + 1;
        return static_cast<double>(base + static_cast<long>(values_.offset()));
    }

    template <typename Array>
    nb::object batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        if (counter_ < 0 && static_cast<std::size_t>(std::max(window_, 1)) <= SMALL_INDEX_SCAN_WINDOW_LIMIT) {
            std::vector<double> min_index(size);
            batch_kernels::high_low_index_small_window_scan(
                input.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                fillna_,
                min_index.data(),
                output.data());
            batch_kernels::rebuild_extreme_state(input.data(), size, static_cast<std::size_t>(std::max(window_, 1)), values_);
            counter_ = static_cast<long>(size) - 1;
        } else {
            batch_kernels::rolling_extreme_index(input.data(), size, values_, fillna_, counter_, output.data());
        }
        return make_array(std::move(output));
    }

private:
    RollingExtreme values_;
    int window_;
    bool fillna_;
    long counter_;
};

class LowIndex {
public:
    LowIndex(int window = 30, bool fillna = true)
        : values_(window, false),
          window_(window),
          fillna_(fillna),
          counter_(-1) {}

    double update(double value) {
        ++counter_;
        values_.push(value);
        if (!fillna_ && !values_.full()) {
            return nan();
        }
        const long base = counter_ - static_cast<long>(values_.size()) + 1;
        return static_cast<double>(base + static_cast<long>(values_.offset()));
    }

    template <typename Array>
    nb::object batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        if (counter_ < 0 && static_cast<std::size_t>(std::max(window_, 1)) <= SMALL_INDEX_SCAN_WINDOW_LIMIT) {
            std::vector<double> max_index(size);
            batch_kernels::high_low_index_small_window_scan(
                input.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                fillna_,
                output.data(),
                max_index.data());
            batch_kernels::rebuild_extreme_state(input.data(), size, static_cast<std::size_t>(std::max(window_, 1)), values_);
            counter_ = static_cast<long>(size) - 1;
        } else {
            batch_kernels::rolling_extreme_index(input.data(), size, values_, fillna_, counter_, output.data());
        }
        return make_array(std::move(output));
    }

private:
    RollingExtreme values_;
    int window_;
    bool fillna_;
    long counter_;
};

class HighLow {
public:
    HighLow(int window = 30, bool fillna = true)
        : min_(window, false),
          max_(window, true),
          window_(window),
          fillna_(fillna),
          last_{nan(), nan()} {}

    HighLowResult update(double value) {
        update_core(value);
        return last_;
    }

    void advance(double value) {
        update_core(value);
    }

    inline const HighLowResult &last() const {
        return last_;
    }

private:
    inline void update_core(double value) {
        min_.push(value);
        max_.push(value);
        if (!fillna_ && !min_.full()) {
            last_ = {nan(), nan()};
        } else {
            last_ = {min_.value(), max_.value()};
        }
    }

public:
    template <typename Array>
    HighLowBatchResult batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> min_values(size);
        std::vector<double> max_values(size);
        if (min_.size() == 0 && static_cast<std::size_t>(std::max(window_, 1)) <= SMALL_SCAN_WINDOW_LIMIT) {
            batch_kernels::high_low_small_window_scan(
                input.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                fillna_,
                min_values.data(),
                max_values.data());
            batch_kernels::rebuild_extreme_state(input.data(), size, static_cast<std::size_t>(std::max(window_, 1)), min_);
            batch_kernels::rebuild_extreme_state(input.data(), size, static_cast<std::size_t>(std::max(window_, 1)), max_);
        } else {
            batch_kernels::high_low(input.data(), size, min_, max_, fillna_, min_values.data(), max_values.data());
        }

        return {make_array(std::move(min_values)), make_array(std::move(max_values))};
    }

private:
    RollingExtreme min_;
    RollingExtreme max_;
    int window_;
    bool fillna_;
    HighLowResult last_;
};

class HighLowIndex {
public:
    HighLowIndex(int window = 30, bool fillna = true)
        : min_(window, false),
          max_(window, true),
          window_(window),
          fillna_(fillna),
          counter_(-1),
          last_{nan(), nan()} {}

    HighLowIndexResult update(double value) {
        update_core(value);
        return last_;
    }

    void advance(double value) {
        update_core(value);
    }

    inline const HighLowIndexResult &last() const {
        return last_;
    }

private:
    inline void update_core(double value) {
        ++counter_;
        min_.push(value);
        max_.push(value);
        if (!fillna_ && !min_.full()) {
            last_ = {nan(), nan()};
        } else {
            const long base = counter_ - static_cast<long>(min_.size()) + 1;
            last_ = {
                static_cast<double>(base + static_cast<long>(min_.offset())),
                static_cast<double>(base + static_cast<long>(max_.offset())),
            };
        }
    }

public:
    template <typename Array>
    HighLowIndexBatchResult batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> min_index(size);
        std::vector<double> max_index(size);
        if (counter_ < 0 && static_cast<std::size_t>(std::max(window_, 1)) <= SMALL_INDEX_SCAN_WINDOW_LIMIT) {
            batch_kernels::high_low_index_small_window_scan(
                input.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                fillna_,
                min_index.data(),
                max_index.data());
            batch_kernels::rebuild_extreme_state(input.data(), size, static_cast<std::size_t>(std::max(window_, 1)), min_);
            batch_kernels::rebuild_extreme_state(input.data(), size, static_cast<std::size_t>(std::max(window_, 1)), max_);
            counter_ = static_cast<long>(size) - 1;
        } else {
            batch_kernels::high_low_index(input.data(), size, min_, max_, fillna_, counter_, min_index.data(), max_index.data());
        }
        return {make_array(std::move(min_index)), make_array(std::move(max_index))};
    }

private:
    RollingExtreme min_;
    RollingExtreme max_;
    int window_;
    bool fillna_;
    long counter_;
    HighLowIndexResult last_;
};

class MidPoint {
public:
    MidPoint(int window = 14, bool fillna = true)
        : window_(static_cast<std::size_t>(std::max(window, 1))),
          min_(window, false),
          max_(window, true),
          history_(window),
          fillna_(fillna) {}

    double update(double value) {
        min_.push(value);
        max_.push(value);
        history_.push(value);
        if (!fillna_ && !min_.full()) {
            return nan();
        }
        return (min_.value() + max_.value()) * 0.5;
    }

    template <typename Array>
    nb::object batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        if (window_ <= SMALL_SCAN_WINDOW_LIMIT) {
            const std::size_t prior_size = history_.size();
            std::vector<double> prior(prior_size);
            for (std::size_t i = 0; i < prior_size; ++i) {
                prior[i] = history_.at(i);
            }
            batch_kernels::midprice_small_window_scan(
                input.data(),
                input.data(),
                size,
                prior.data(),
                prior.data(),
                prior_size,
                window_,
                fillna_,
                output.data());
            rebuild_state(input.data(), size, prior);
        } else {
            batch_kernels::midpoint(input.data(), size, min_, max_, fillna_, output.data());
        }
        return make_array(std::move(output));
    }

private:
    template <typename Value>
    void rebuild_state(const Value *values, std::size_t size, const std::vector<double> &prior) {
        const std::size_t prior_size = prior.size();
        const std::size_t total = prior_size + size;
        const std::size_t start = total > window_ ? total - window_ : 0;

        min_.reset();
        max_.reset();
        history_.reset();
        for (std::size_t index = start; index < total; ++index) {
            const double value = index < prior_size ? prior[index] : static_cast<double>(values[index - prior_size]);
            min_.push(value);
            max_.push(value);
            history_.push(value);
        }
    }

    std::size_t window_;
    RollingExtreme min_;
    RollingExtreme max_;
    RollingBuffer history_;
    bool fillna_;
};

class MidPrice {
public:
    MidPrice(int window = 14, bool fillna = true)
        : window_(static_cast<std::size_t>(std::max(window, 1))),
          highs_(window, true),
          lows_(window, false),
          high_history_(window),
          low_history_(window),
          fillna_(fillna) {}

    double update(double high, double low) {
        highs_.push(high);
        lows_.push(low);
        high_history_.push(high);
        low_history_.push(low);
        if (!fillna_ && !highs_.full()) {
            return nan();
        }
        return (highs_.value() + lows_.value()) * 0.5;
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &high, const Array1 &low) {
        const std::size_t size = high.shape(0);
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        if (window_ <= small_scan_window_limit_) {
            const std::size_t prior_size = high_history_.size();
            std::vector<double> prior_high(prior_size);
            std::vector<double> prior_low(prior_size);
            for (std::size_t i = 0; i < prior_size; ++i) {
                prior_high[i] = high_history_.at(i);
                prior_low[i] = low_history_.at(i);
            }

            batch_kernels::midprice_small_window_scan(
                high.data(),
                low.data(),
                size,
                prior_high.data(),
                prior_low.data(),
                prior_size,
                window_,
                fillna_,
                output.data());
            rebuild_state(high.data(), low.data(), size, prior_high, prior_low);
        } else {
            batch_kernels::midprice(high.data(), low.data(), size, highs_, lows_, fillna_, output.data());
            for (std::size_t i = 0; i < size; ++i) {
                high_history_.push(static_cast<double>(high.data()[i]));
                low_history_.push(static_cast<double>(low.data()[i]));
            }
        }
        return make_array(std::move(output));
    }

private:
    template <typename High, typename Low>
    void rebuild_state(
        const High *high,
        const Low *low,
        std::size_t size,
        const std::vector<double> &prior_high,
        const std::vector<double> &prior_low) {
        const std::size_t prior_size = prior_high.size();
        const std::size_t total = prior_size + size;
        const std::size_t start = total > window_ ? total - window_ : 0;

        highs_.reset();
        lows_.reset();
        high_history_.reset();
        low_history_.reset();
        for (std::size_t index = start; index < total; ++index) {
            const double high_value = index < prior_size ? prior_high[index] : static_cast<double>(high[index - prior_size]);
            const double low_value = index < prior_size ? prior_low[index] : static_cast<double>(low[index - prior_size]);
            highs_.push(high_value);
            lows_.push(low_value);
            high_history_.push(high_value);
            low_history_.push(low_value);
        }
    }

    static constexpr std::size_t small_scan_window_limit_ = 64;
    std::size_t window_;
    RollingExtreme highs_;
    RollingExtreme lows_;
    RollingBuffer high_history_;
    RollingBuffer low_history_;
    bool fillna_;
};

class DonchianChannel {
public:
    DonchianChannel(int window = 20, bool fillna = true)
        : highs_(window, true),
          lows_(window, false),
          close_(window),
          close_sum_(0.0),
          fillna_(fillna),
          last_{nan(), nan(), nan(), nan(), nan()} {}

    DonchianChannelResult update(double close, double high, double low) {
        update_core(close, high, low);
        return last_;
    }

    void advance(double close, double high, double low) {
        update_core(close, high, low);
    }

    inline const DonchianChannelResult &last() const {
        return last_;
    }

private:
    inline void update_core(double close, double high, double low) {
        highs_.push(high);
        lows_.push(low);
        if (close_.full()) {
            close_sum_ -= close_.oldest();
        }
        close_.push(close);
        close_sum_ += close;

        if (!fillna_ && !highs_.full()) {
            last_ = {nan(), nan(), nan(), nan(), nan()};
            return;
        }

        const double upper = highs_.value();
        const double lower = lows_.value();
        const double middle = (upper + lower) * 0.5;
        const double close_mean = close_sum_ / close_.size();
        last_ = {
            upper,
            lower,
            middle,
            safe_divide((upper - lower) * 100.0, close_mean),
            safe_divide(close - lower, upper - lower),
        };
    }

public:
    DonchianChannelBatchResult batch(const InputArray &close, const InputArray &high, const InputArray &low) {
        const std::size_t size = close.shape(0);
        std::vector<double> upper(size);
        std::vector<double> lower(size);
        std::vector<double> middle(size);
        std::vector<double> width(size);
        std::vector<double> percent(size);

        for (std::size_t i = 0; i < size; ++i) {
            const DonchianChannelResult out = update(close(i), high(i), low(i));
            upper[i] = out.upper;
            lower[i] = out.lower;
            middle[i] = out.middle;
            width[i] = out.width;
            percent[i] = out.percent;
        }

        return {
            make_array(std::move(upper)),
            make_array(std::move(lower)),
            make_array(std::move(middle)),
            make_array(std::move(width)),
            make_array(std::move(percent)),
        };
    }

private:
    RollingExtreme highs_;
    RollingExtreme lows_;
    RollingBuffer close_;
    double close_sum_;
    bool fillna_;
    DonchianChannelResult last_;
};

class FibonacciRetracementLevels {
public:
    FibonacciRetracementLevels(int window = 30, bool uptrend = true, bool fillna = true)
        : highs_(window, true),
          lows_(window, false),
          fillna_(fillna),
          uptrend_(uptrend),
          last_{nan(), nan(), nan(), nan(), nan(), nan()} {}

    FibonacciRetracementLevelsResult update(double high, double low) {
        update_core(high, low);
        return last_;
    }

    void advance(double high, double low) {
        update_core(high, low);
    }

    inline const FibonacciRetracementLevelsResult &last() const {
        return last_;
    }

    template <typename Array0, typename Array1>
    FibonacciRetracementLevelsBatchResult batch_array(const Array0 &high, const Array1 &low) {
        const std::size_t size = high.shape(0);
        require_same_size(size, low.shape(0));
        std::vector<double> level0(size);
        std::vector<double> level236(size);
        std::vector<double> level382(size);
        std::vector<double> level500(size);
        std::vector<double> level618(size);
        std::vector<double> level100(size);
        const auto *high_values = high.data();
        const auto *low_values = low.data();

        for (std::size_t i = 0; i < size; ++i) {
            const FibonacciRetracementLevelsResult out = update(
                static_cast<double>(high_values[i]),
                static_cast<double>(low_values[i]));
            level0[i] = out.level0;
            level236[i] = out.level236;
            level382[i] = out.level382;
            level500[i] = out.level500;
            level618[i] = out.level618;
            level100[i] = out.level100;
        }

        return {
            make_array(std::move(level0)),
            make_array(std::move(level236)),
            make_array(std::move(level382)),
            make_array(std::move(level500)),
            make_array(std::move(level618)),
            make_array(std::move(level100)),
        };
    }

private:
    inline void update_core(double high, double low) {
        highs_.push(high);
        lows_.push(low);

        if (!fillna_ && !highs_.full()) {
            last_ = {nan(), nan(), nan(), nan(), nan(), nan()};
            return;
        }

        const double highest = highs_.value();
        const double lowest = lows_.value();
        const double range = highest - lowest;
        if (uptrend_) {
            last_ = {
                highest,
                highest - 0.236 * range,
                highest - 0.382 * range,
                highest - 0.5 * range,
                highest - 0.618 * range,
                lowest,
            };
        } else {
            last_ = {
                lowest,
                lowest + 0.236 * range,
                lowest + 0.382 * range,
                lowest + 0.5 * range,
                lowest + 0.618 * range,
                highest,
            };
        }
    }

    RollingExtreme highs_;
    RollingExtreme lows_;
    bool fillna_;
    bool uptrend_;
    FibonacciRetracementLevelsResult last_;
};

class UlcerIndex {
public:
    UlcerIndex(int window = 14, bool fillna = true)
        : close_(window, true),
          drawdowns_(window),
          window_(window),
          fillna_(fillna),
          drawdown_sum2_(0.0) {}

    double update(double close) {
        close_.push(close);
        const double max_close = close_.value();
        const double drawdown = safe_divide(100.0 * (close - max_close), max_close);
        if (drawdowns_.full()) {
            const double oldest = drawdowns_.oldest();
            drawdown_sum2_ -= oldest * oldest;
        }
        drawdowns_.push(drawdown);
        drawdown_sum2_ += drawdown * drawdown;

        if (!fillna_ && !drawdowns_.full()) {
            return nan();
        }

        return std::sqrt(drawdown_sum2_ / window_);
    }

    nb::object batch(const InputArray &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(close(i));
        }
        return make_array(std::move(output));
    }

private:
    RollingExtreme close_;
    RollingBuffer drawdowns_;
    int window_;
    bool fillna_;
    double drawdown_sum2_;
};

class Aroon {
public:
    Aroon(int window = 14, bool fillna = true)
        : highs_(window + 1, true),
          lows_(window + 1, false),
          window_(window),
          fillna_(fillna),
          last_{nan(), nan()} {}

    AroonResult update(double high, double low) {
        update_core(high, low);
        return last_;
    }

    void advance(double high, double low) {
        update_core(high, low);
    }

    inline const AroonResult &last() const {
        return last_;
    }

private:
    inline void update_core(double high, double low) {
        highs_.push(high);
        lows_.push(low);
        if (!fillna_ && !highs_.full()) {
            last_ = {nan(), nan()};
            return;
        }

        const double denom = static_cast<double>(std::max<std::size_t>(highs_.size() - 1, 1));
        const double periods_since_high = denom - static_cast<double>(highs_.offset());
        const double periods_since_low = denom - static_cast<double>(lows_.offset());

        last_ = {
            100.0 * (denom - periods_since_low) / denom,
            100.0 * (denom - periods_since_high) / denom,
        };
    }

public:
    template <typename Array0, typename Array1>
    AroonBatchResult batch_array(const Array0 &high, const Array1 &low) {
        const std::size_t size = high.shape(0);
        require_same_size(size, low.shape(0));
        std::vector<double> down(size);
        std::vector<double> up(size);
        if (highs_.size() == 0 && static_cast<std::size_t>(std::max(window_, 1)) <= SMALL_INDEX_SCAN_WINDOW_LIMIT) {
            batch_kernels::aroon_small_window_scan(
                high.data(),
                low.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                fillna_,
                down.data(),
                up.data());
            batch_kernels::rebuild_pair_extreme_state(
                high.data(),
                low.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)) + 1,
                highs_,
                lows_);
        } else {
            batch_kernels::aroon(high.data(), low.data(), size, highs_, lows_, fillna_, down.data(), up.data());
        }
        return {make_array(std::move(down)), make_array(std::move(up))};
    }

    template <typename Array0, typename Array1>
    nb::object batch_oscillator_array(const Array0 &high, const Array1 &low) {
        const std::size_t size = high.shape(0);
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        if (highs_.size() == 0 && static_cast<std::size_t>(std::max(window_, 1)) <= SMALL_INDEX_SCAN_WINDOW_LIMIT) {
            std::vector<double> down(size);
            batch_kernels::aroon_small_window_scan(
                high.data(),
                low.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                fillna_,
                down.data(),
                output.data());
            for (std::size_t i = 0; i < size; ++i) {
                output[i] = (std::isnan(output[i]) || std::isnan(down[i])) ? nan() : output[i] - down[i];
            }
            batch_kernels::rebuild_pair_extreme_state(
                high.data(),
                low.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)) + 1,
                highs_,
                lows_);
        } else {
            batch_kernels::aroon_oscillator(high.data(), low.data(), size, highs_, lows_, fillna_, output.data());
        }
        return make_array(std::move(output));
    }

private:
    RollingExtreme highs_;
    RollingExtreme lows_;
    int window_;
    bool fillna_;
    AroonResult last_;
};

class AroonOscillator {
public:
    AroonOscillator(int window = 14, bool fillna = true)
        : aroon_(window, fillna) {}

    double update(double high, double low) {
        const AroonResult out = aroon_.update(high, low);
        return out.up - out.down;
    }

    void advance(double high, double low) {
        aroon_.advance(high, low);
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &high, const Array1 &low) {
        return aroon_.batch_oscillator_array(high, low);
    }

private:
    Aroon aroon_;
};

class Vortex {
public:
    Vortex(int window = 14, bool fillna = true)
        : true_range_(window),
          positive_movement_(window),
          negative_movement_(window),
          previous_high_(0.0),
          previous_low_(0.0),
          previous_close_(0.0),
          first_(true),
          fillna_(fillna),
          last_{nan(), nan(), nan()} {}

    VortexResult update(double close, double high, double low) {
        update_core(close, high, low);
        return last_;
    }

    void advance(double close, double high, double low) {
        update_core(close, high, low);
    }

    inline const VortexResult &last() const {
        return last_;
    }

private:
    inline void update_core(double close, double high, double low) {
        const double tr = first_ ? high - low : true_range(close, high, low, previous_close_);
        const double positive = first_ ? 0.0 : std::abs(high - previous_low_);
        const double negative = first_ ? 0.0 : std::abs(low - previous_high_);

        previous_high_ = high;
        previous_low_ = low;
        previous_close_ = close;
        first_ = false;

        true_range_.push(tr);
        positive_movement_.push(positive);
        negative_movement_.push(negative);

        if (!fillna_ && !true_range_.full()) {
            last_ = {nan(), nan(), nan()};
            return;
        }

        const double true_range_sum = true_range_.sum();
        const double positive_indicator = safe_divide(positive_movement_.sum(), true_range_sum, 1.0);
        const double negative_indicator = safe_divide(negative_movement_.sum(), true_range_sum, 1.0);
        last_ = {positive_indicator, negative_indicator, positive_indicator - negative_indicator};
    }

public:
    VortexBatchResult batch(const InputArray &close, const InputArray &high, const InputArray &low) {
        const std::size_t size = close.shape(0);
        std::vector<double> positive(size);
        std::vector<double> negative(size);
        std::vector<double> difference(size);

        for (std::size_t i = 0; i < size; ++i) {
            const VortexResult out = update(close(i), high(i), low(i));
            positive[i] = out.positive;
            negative[i] = out.negative;
            difference[i] = out.difference;
        }

        return {make_array(std::move(positive)), make_array(std::move(negative)), make_array(std::move(difference))};
    }

private:
    RollingSumWindow true_range_;
    RollingSumWindow positive_movement_;
    RollingSumWindow negative_movement_;
    double previous_high_;
    double previous_low_;
    double previous_close_;
    bool first_;
    bool fillna_;
    VortexResult last_;
};

class DetrendedPriceOscillator {
public:
    DetrendedPriceOscillator(int window = 20, bool fillna = true)
        : shifted_close_(static_cast<int>(0.5 * window) + 1, true),
          mean_(window, fillna),
          window_(window),
          shift_(static_cast<int>(0.5 * window) + 1),
          fillna_(fillna),
          counter_(0) {}

    double update(double close) {
        const double shifted = shifted_close_.update(close);
        const double mean = mean_.update(close);
        const bool return_nan = !fillna_ && counter_ < std::max(window_, shift_);
        ++counter_;
        return return_nan ? nan() : shifted - mean;
    }

    nb::object batch(const InputArray &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(close(i));
        }
        return make_array(std::move(output));
    }

private:
    Delay shifted_close_;
    SMA mean_;
    int window_;
    int shift_;
    bool fillna_;
    long counter_;
};

class KSTOscillator {
public:
    KSTOscillator(
        int roc1 = 10,
        int roc2 = 15,
        int roc3 = 20,
        int roc4 = 30,
        int window1 = 10,
        int window2 = 10,
        int window3 = 10,
        int window4 = 15,
        int signal = 9,
        bool fillna = true)
        : delay1_(roc1, true),
          delay2_(roc2, true),
          delay3_(roc3, true),
          delay4_(roc4, true),
          mean1_(window1, fillna),
          mean2_(window2, fillna),
          mean3_(window3, fillna),
          mean4_(window4, fillna),
          signal_(signal, fillna),
          ready_(std::max({roc1 + window1, roc2 + window2, roc3 + window3, roc4 + window4, signal})),
          fillna_(fillna),
          counter_(0),
          last_{nan(), nan(), nan()} {}

    KSTOscillatorResult update(double close) {
        update_core(close);
        return last_;
    }

    void advance(double close) {
        update_core(close);
    }

    inline const KSTOscillatorResult &last() const {
        return last_;
    }

private:
    inline void update_core(double close) {
        const double roc1 = roc(delay1_.update(close), close);
        const double roc2 = roc(delay2_.update(close), close);
        const double roc3 = roc(delay3_.update(close), close);
        const double roc4 = roc(delay4_.update(close), close);
        const double kst =
            100.0 * (mean1_.update(roc1) + 2.0 * mean2_.update(roc2) + 3.0 * mean3_.update(roc3) + 4.0 * mean4_.update(roc4));
        const double signal = signal_.update(kst);
        const bool return_nan = !fillna_ && counter_ < ready_;
        ++counter_;

        last_ = {
            return_nan ? nan() : kst,
            return_nan ? nan() : signal,
            return_nan ? nan() : kst - signal,
        };
    }

public:
    KSTOscillatorBatchResult batch(const InputArray &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> kst(size);
        std::vector<double> signal(size);
        std::vector<double> difference(size);
        for (std::size_t i = 0; i < size; ++i) {
            const KSTOscillatorResult out = update(close(i));
            kst[i] = out.kst;
            signal[i] = out.signal;
            difference[i] = out.difference;
        }

        return {make_array(std::move(kst)), make_array(std::move(signal)), make_array(std::move(difference))};
    }

private:
    static double roc(double previous, double close) {
        return previous == 0.0 ? 0.0 : (close - previous) / previous;
    }

    Delay delay1_;
    Delay delay2_;
    Delay delay3_;
    Delay delay4_;
    SMA mean1_;
    SMA mean2_;
    SMA mean3_;
    SMA mean4_;
    SMA signal_;
    int ready_;
    bool fillna_;
    long counter_;
    KSTOscillatorResult last_;
};

class Ichimoku {
public:
    Ichimoku(int window1 = 9, int window2 = 26, int window3 = 52, bool fillna = true)
        : highs1_(window1, true),
          lows1_(window1, false),
          highs2_(window2, true),
          lows2_(window2, false),
          highs3_(window3, true),
          lows3_(window3, false),
          fillna_(fillna),
          last_{nan(), nan(), nan(), nan()} {}

    IchimokuResult update(double high, double low) {
        update_core(high, low);
        return last_;
    }

    void advance(double high, double low) {
        update_core(high, low);
    }

    inline const IchimokuResult &last() const {
        return last_;
    }

private:
    inline void update_core(double high, double low) {
        highs1_.push(high);
        lows1_.push(low);
        highs2_.push(high);
        lows2_.push(low);
        highs3_.push(high);
        lows3_.push(low);

        const double conversion = (!fillna_ && !highs1_.full()) ? nan() : (highs1_.value() + lows1_.value()) * 0.5;
        const double base = (!fillna_ && !highs2_.full()) ? nan() : (highs2_.value() + lows2_.value()) * 0.5;
        const double span_a = (std::isnan(conversion) || std::isnan(base)) ? nan() : (conversion + base) * 0.5;
        const double span_b = (!fillna_ && !highs3_.full()) ? nan() : (highs3_.value() + lows3_.value()) * 0.5;

        last_ = {conversion, base, span_a, span_b};
    }

public:
    IchimokuBatchResult batch(const InputArray &high, const InputArray &low) {
        const std::size_t size = high.shape(0);
        std::vector<double> conversion(size);
        std::vector<double> base(size);
        std::vector<double> span_a(size);
        std::vector<double> span_b(size);
        for (std::size_t i = 0; i < size; ++i) {
            const IchimokuResult out = update(high(i), low(i));
            conversion[i] = out.conversion;
            base[i] = out.base;
            span_a[i] = out.span_a;
            span_b[i] = out.span_b;
        }

        return {
            make_array(std::move(conversion)),
            make_array(std::move(base)),
            make_array(std::move(span_a)),
            make_array(std::move(span_b)),
        };
    }

private:
    RollingExtreme highs1_;
    RollingExtreme lows1_;
    RollingExtreme highs2_;
    RollingExtreme lows2_;
    RollingExtreme highs3_;
    RollingExtreme lows3_;
    bool fillna_;
    IchimokuResult last_;
};

struct DirectionalValues {
    double plus_dm;
    double minus_dm;
    double tr;
    double plus_di;
    double minus_di;
    double dx;
};

class DirectionalMovementCore {
public:
    explicit DirectionalMovementCore(int window = 14)
        : plus_dm_(window),
          minus_dm_(window),
          tr_(window),
          previous_high_(0.0),
          previous_low_(0.0),
          previous_close_(0.0),
          first_(true) {}

    DirectionalValues update(double close, double high, double low) {
        double plus = 0.0;
        double minus = 0.0;
        double tr = high - low;

        if (!first_) {
            const double up_move = high - previous_high_;
            const double down_move = previous_low_ - low;
            plus = (up_move > down_move && up_move > 0.0) ? up_move : 0.0;
            minus = (down_move > up_move && down_move > 0.0) ? down_move : 0.0;
            tr = true_range(close, high, low, previous_close_);
        }

        previous_high_ = high;
        previous_low_ = low;
        previous_close_ = close;
        first_ = false;

        const double smoothed_plus = plus_dm_.update(plus);
        const double smoothed_minus = minus_dm_.update(minus);
        const double smoothed_tr = tr_.update(tr);
        const double plus_di = safe_divide(100.0 * smoothed_plus, smoothed_tr);
        const double minus_di = safe_divide(100.0 * smoothed_minus, smoothed_tr);
        const double dx = safe_divide(100.0 * std::abs(plus_di - minus_di), plus_di + minus_di);

        return {smoothed_plus, smoothed_minus, smoothed_tr, plus_di, minus_di, dx};
    }

    template <typename Close, typename High, typename Low>
    void batch_indicator(
        const Close *close,
        const High *high,
        const Low *low,
        std::size_t size,
        int selector,
        double *output) {
        const int window = tr_.window();
        double plus_value = plus_dm_.value();
        double minus_value = minus_dm_.value();
        double tr_value = tr_.value();
        long plus_count = plus_dm_.count();
        long minus_count = minus_dm_.count();
        long tr_count = tr_.count();
        double previous_high = previous_high_;
        double previous_low = previous_low_;
        double previous_close = previous_close_;
        bool first = first_;

        for (std::size_t i = 0; i < size; ++i) {
            const double close_value = static_cast<double>(close[i]);
            const double high_value = static_cast<double>(high[i]);
            const double low_value = static_cast<double>(low[i]);
            double plus = 0.0;
            double minus = 0.0;
            double tr = high_value - low_value;

            if (!first) {
                const double up_move = high_value - previous_high;
                const double down_move = previous_low - low_value;
                plus = (up_move > down_move && up_move > 0.0) ? up_move : 0.0;
                minus = (down_move > up_move && down_move > 0.0) ? down_move : 0.0;
                tr = true_range(close_value, high_value, low_value, previous_close);
            }

            previous_high = high_value;
            previous_low = low_value;
            previous_close = close_value;
            first = false;

            plus_value = (plus_value * (window - 1.0) + plus) / window;
            minus_value = (minus_value * (window - 1.0) + minus) / window;
            tr_value = (tr_value * (window - 1.0) + tr) / window;
            ++plus_count;
            ++minus_count;
            ++tr_count;

            const double plus_di = safe_divide(100.0 * plus_value, tr_value);
            const double minus_di = safe_divide(100.0 * minus_value, tr_value);
            if (selector == 0) {
                output[i] = plus_di;
            } else if (selector == 1) {
                output[i] = minus_di;
            } else {
                output[i] = safe_divide(100.0 * std::abs(plus_di - minus_di), plus_di + minus_di);
            }
        }

        previous_high_ = previous_high;
        previous_low_ = previous_low;
        previous_close_ = previous_close;
        first_ = first;
        plus_dm_.set_state(plus_value, plus_count);
        minus_dm_.set_state(minus_value, minus_count);
        tr_.set_state(tr_value, tr_count);
    }

private:
    WilderSmoothing plus_dm_;
    WilderSmoothing minus_dm_;
    WilderSmoothing tr_;
    double previous_high_;
    double previous_low_;
    double previous_close_;
    bool first_;
};

namespace batch_kernels {

template <typename Close, typename High, typename Low>
void plus_directional_indicator(
    const Close *close,
    const High *high,
    const Low *low,
    std::size_t size,
    DirectionalMovementCore &core,
    int window,
    bool fillna,
    long &counter,
    double *output) {
    core.batch_indicator(close, high, low, size, 0, output);
    for (std::size_t i = 0; i < size; ++i) {
        const bool return_nan = !fillna && counter < window;
        ++counter;
        if (return_nan) {
            output[i] = nan();
        }
    }
}

template <typename Close, typename High, typename Low>
void minus_directional_indicator(
    const Close *close,
    const High *high,
    const Low *low,
    std::size_t size,
    DirectionalMovementCore &core,
    int window,
    bool fillna,
    long &counter,
    double *output) {
    core.batch_indicator(close, high, low, size, 1, output);
    for (std::size_t i = 0; i < size; ++i) {
        const bool return_nan = !fillna && counter < window;
        ++counter;
        if (return_nan) {
            output[i] = nan();
        }
    }
}

template <typename Close, typename High, typename Low>
void directional_movement_index(
    const Close *close,
    const High *high,
    const Low *low,
    std::size_t size,
    DirectionalMovementCore &core,
    int window,
    bool fillna,
    long &counter,
    double *output) {
    core.batch_indicator(close, high, low, size, 2, output);
    for (std::size_t i = 0; i < size; ++i) {
        const bool return_nan = !fillna && counter < window;
        ++counter;
        if (return_nan) {
            output[i] = nan();
        }
    }
}

}  // namespace batch_kernels

class PlusDirectionalMovement {
public:
    PlusDirectionalMovement(int window = 14, bool fillna = true)
        : smoothing_(window),
          previous_high_(0.0),
          previous_low_(0.0),
          first_(true),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double high, double low) {
        double plus = 0.0;
        if (!first_) {
            const double up_move = high - previous_high_;
            const double down_move = previous_low_ - low;
            plus = (up_move > down_move && up_move > 0.0) ? up_move : 0.0;
        }
        previous_high_ = high;
        previous_low_ = low;
        first_ = false;
        const double retval = smoothing_.update(plus);
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : retval;
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &high, const Array1 &low) {
        const std::size_t size = high.shape(0);
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        const int window = smoothing_.window();
        double smoothed = smoothing_.value();
        long smooth_count = smoothing_.count();
        double previous_high = previous_high_;
        double previous_low = previous_low_;
        bool first = first_;
        long counter = counter_;

        for (std::size_t i = 0; i < size; ++i) {
            const double high_value = static_cast<double>(high_values[i]);
            const double low_value = static_cast<double>(low_values[i]);
            double plus = 0.0;
            if (!first) {
                const double up_move = high_value - previous_high;
                const double down_move = previous_low - low_value;
                plus = (up_move > down_move && up_move > 0.0) ? up_move : 0.0;
            }
            previous_high = high_value;
            previous_low = low_value;
            first = false;
            smoothed = (smoothed * (window - 1.0) + plus) / window;
            ++smooth_count;
            const bool return_nan = !fillna_ && counter < window_;
            ++counter;
            output[i] = return_nan ? nan() : smoothed;
        }

        previous_high_ = previous_high;
        previous_low_ = previous_low;
        first_ = first;
        counter_ = counter;
        smoothing_.set_state(smoothed, smooth_count);
        return make_array(std::move(output));
    }

private:
    WilderSmoothing smoothing_;
    double previous_high_;
    double previous_low_;
    bool first_;
    int window_;
    bool fillna_;
    long counter_;
};

class MinusDirectionalMovement {
public:
    MinusDirectionalMovement(int window = 14, bool fillna = true)
        : smoothing_(window),
          previous_high_(0.0),
          previous_low_(0.0),
          first_(true),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double high, double low) {
        double minus = 0.0;
        if (!first_) {
            const double up_move = high - previous_high_;
            const double down_move = previous_low_ - low;
            minus = (down_move > up_move && down_move > 0.0) ? down_move : 0.0;
        }
        previous_high_ = high;
        previous_low_ = low;
        first_ = false;
        const double retval = smoothing_.update(minus);
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : retval;
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &high, const Array1 &low) {
        const std::size_t size = high.shape(0);
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        const int window = smoothing_.window();
        double smoothed = smoothing_.value();
        long smooth_count = smoothing_.count();
        double previous_high = previous_high_;
        double previous_low = previous_low_;
        bool first = first_;
        long counter = counter_;

        for (std::size_t i = 0; i < size; ++i) {
            const double high_value = static_cast<double>(high_values[i]);
            const double low_value = static_cast<double>(low_values[i]);
            double minus = 0.0;
            if (!first) {
                const double up_move = high_value - previous_high;
                const double down_move = previous_low - low_value;
                minus = (down_move > up_move && down_move > 0.0) ? down_move : 0.0;
            }
            previous_high = high_value;
            previous_low = low_value;
            first = false;
            smoothed = (smoothed * (window - 1.0) + minus) / window;
            ++smooth_count;
            const bool return_nan = !fillna_ && counter < window_;
            ++counter;
            output[i] = return_nan ? nan() : smoothed;
        }

        previous_high_ = previous_high;
        previous_low_ = previous_low;
        first_ = first;
        counter_ = counter;
        smoothing_.set_state(smoothed, smooth_count);
        return make_array(std::move(output));
    }

private:
    WilderSmoothing smoothing_;
    double previous_high_;
    double previous_low_;
    bool first_;
    int window_;
    bool fillna_;
    long counter_;
};

class PlusDirectionalIndicator {
public:
    PlusDirectionalIndicator(int window = 14, bool fillna = true)
        : core_(window),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double close, double high, double low) {
        const double retval = core_.update(close, high, low).plus_di;
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : retval;
    }

    template <typename Array0, typename Array1, typename Array2>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        batch_kernels::plus_directional_indicator(
            close.data(),
            high.data(),
            low.data(),
            size,
            core_,
            window_,
            fillna_,
            counter_,
            output.data());
        return make_array(std::move(output));
    }

private:
    DirectionalMovementCore core_;
    int window_;
    bool fillna_;
    long counter_;
};

class MinusDirectionalIndicator {
public:
    MinusDirectionalIndicator(int window = 14, bool fillna = true)
        : core_(window),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double close, double high, double low) {
        const double retval = core_.update(close, high, low).minus_di;
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : retval;
    }

    template <typename Array0, typename Array1, typename Array2>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        batch_kernels::minus_directional_indicator(
            close.data(),
            high.data(),
            low.data(),
            size,
            core_,
            window_,
            fillna_,
            counter_,
            output.data());
        return make_array(std::move(output));
    }

private:
    DirectionalMovementCore core_;
    int window_;
    bool fillna_;
    long counter_;
};

class DirectionalMovementIndex {
public:
    DirectionalMovementIndex(int window = 14, bool fillna = true)
        : core_(window),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double close, double high, double low) {
        const double retval = core_.update(close, high, low).dx;
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : retval;
    }

    template <typename Array0, typename Array1, typename Array2>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        batch_kernels::directional_movement_index(
            close.data(),
            high.data(),
            low.data(),
            size,
            core_,
            window_,
            fillna_,
            counter_,
            output.data());
        return make_array(std::move(output));
    }

private:
    DirectionalMovementCore core_;
    int window_;
    bool fillna_;
    long counter_;
};

class AverageDirectionalMovementIndex {
public:
    AverageDirectionalMovementIndex(int window = 14, bool fillna = true)
        : core_(window),
          adx_(window),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double close, double high, double low) {
        const double retval = adx_.update(core_.update(close, high, low).dx);
        const bool return_nan = !fillna_ && counter_ < window_ * 2;
        ++counter_;
        return return_nan ? nan() : retval;
    }

    template <typename Array0, typename Array1, typename Array2>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        core_.batch_indicator(close.data(), high.data(), low.data(), size, 2, output.data());

        const int window = adx_.window();
        double adx_value = adx_.value();
        long adx_count = adx_.count();
        long counter = counter_;
        for (std::size_t i = 0; i < size; ++i) {
            adx_value = (adx_value * (window - 1.0) + output[i]) / window;
            ++adx_count;
            const bool return_nan = !fillna_ && counter < window_ * 2;
            ++counter;
            output[i] = return_nan ? nan() : adx_value;
        }

        adx_.set_state(adx_value, adx_count);
        counter_ = counter;
        return make_array(std::move(output));
    }

private:
    DirectionalMovementCore core_;
    WilderSmoothing adx_;
    int window_;
    bool fillna_;
    long counter_;
};

class AverageDirectionalMovementIndexRating {
public:
    AverageDirectionalMovementIndexRating(int window = 14, bool fillna = true)
        : adx_(window, fillna),
          delayed_(window, fillna),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double close, double high, double low) {
        const double adx = adx_.update(close, high, low);
        const double previous = delayed_.update(adx);
        const bool return_nan = !fillna_ && counter_ < window_ * 3;
        ++counter_;
        return return_nan ? nan() : (adx + previous) * 0.5;
    }

private:
    AverageDirectionalMovementIndex adx_;
    Delay delayed_;
    int window_;
    bool fillna_;
    long counter_;
};

class PercentagePrice {
public:
    PercentagePrice(double window_1 = 12.0, double window_2 = 26.0, double window_3 = 9.0, bool fillna = false)
        : oscillator_1_(window_1, true),
          oscillator_2_(window_2, true),
          oscillator_3_(window_3, true),
          window_(fillna ? 0 : static_cast<int>(std::max({window_1, window_2, window_3}))),
          counter_(0),
          last_{nan(), nan(), nan()} {}

    PercentagePriceResult update(double close) {
        return update_value(close);
    }

    void advance(double close) {
        update_core(close);
    }

    inline const PercentagePriceResult &last() const {
        return last_;
    }

    PercentagePriceBatchResult batch(const InputArray &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> ppo(size);
        std::vector<double> signal(size);
        std::vector<double> histogram(size);
        const double *values = close.data();

        for (std::size_t i = 0; i < size; ++i) {
            const PercentagePriceResult output = update_value(values[i]);
            ppo[i] = output.ppo;
            signal[i] = output.signal;
            histogram[i] = output.histogram;
        }

        return {make_array(std::move(ppo)), make_array(std::move(signal)), make_array(std::move(histogram))};
    }

    PercentagePriceBatchResult batch_records(nb::iterable records) {
        if (table_has_column(records, "close")) {
            nb::object close = table_column_array(records, "close");
            switch (array_dtype(close)) {
                case InputDType::Float32: {
                    const FloatInputArray input = nb::cast<FloatInputArray>(close);
                    const std::size_t size = input.shape(0);
                    std::vector<double> ppo(size);
                    std::vector<double> signal(size);
                    std::vector<double> histogram(size);
                    const float *values = input.data();
                    for (std::size_t i = 0; i < size; ++i) {
                        const PercentagePriceResult out = update_value(static_cast<double>(values[i]));
                        ppo[i] = out.ppo;
                        signal[i] = out.signal;
                        histogram[i] = out.histogram;
                    }
                    return {make_array(std::move(ppo)), make_array(std::move(signal)), make_array(std::move(histogram))};
                }
                case InputDType::Float64:
                    return batch(nb::cast<InputArray>(close));
            }
        }

        std::vector<double> ppo = make_record_output(records);
        std::vector<double> signal;
        std::vector<double> histogram;
        signal.reserve(ppo.capacity());
        histogram.reserve(ppo.capacity());

        for (nb::handle record : records) {
            const PercentagePriceResult output = update_value(record_value(record, "close", 0));
            ppo.push_back(output.ppo);
            signal.push_back(output.signal);
            histogram.push_back(output.histogram);
        }

        return {make_array(std::move(ppo)), make_array(std::move(signal)), make_array(std::move(histogram))};
    }

    nb::object batch_ppo(const InputArray &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        const double *values = close.data();

        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update_value(values[i]).ppo;
        }

        return make_array(std::move(output));
    }

    nb::object batch_ppo_records(nb::iterable records) {
        if (table_has_column(records, "close")) {
            nb::object close = table_column_array(records, "close");
            switch (array_dtype(close)) {
                case InputDType::Float32: {
                    const FloatInputArray input = nb::cast<FloatInputArray>(close);
                    const std::size_t size = input.shape(0);
                    std::vector<double> output(size);
                    const float *values = input.data();
                    for (std::size_t i = 0; i < size; ++i) {
                        output[i] = update_value(static_cast<double>(values[i])).ppo;
                    }
                    return make_array(std::move(output));
                }
                case InputDType::Float64:
                    return batch_ppo(nb::cast<InputArray>(close));
            }
        }

        std::vector<double> output = make_record_output(records);
        for (nb::handle record : records) {
            output.push_back(update_value(record_value(record, "close", 0)).ppo);
        }
        return make_array(std::move(output));
    }

private:
    PercentagePriceResult update_value(double close) {
        update_core(close);
        return last_;
    }

    inline void update_core(double close) {
        ++counter_;

        const double o26 = oscillator_2_.update(close);
        const double ppo = ((oscillator_1_.update(close) - o26) / o26) * 100.0;
        const double signal = oscillator_3_.update(ppo);
        const double histogram = ppo - signal;

        last_ = {ppo, signal, histogram};
    }

    EMA oscillator_1_;
    EMA oscillator_2_;
    EMA oscillator_3_;
    int window_;
    long counter_;
    PercentagePriceResult last_;
};

class PercentageVolume {
public:
    PercentageVolume(double window_1 = 12.0, double window_2 = 26.0, double signal = 9.0, bool fillna = true)
        : oscillator_1_(window_1, true),
          oscillator_2_(window_2, true),
          signal_(signal, true),
          counter_(fillna ? 0 : -static_cast<long>(std::max(window_1, window_2))),
          last_{nan(), nan(), nan()} {}

    PercentageVolumeResult update(double volume) {
        update_core(volume);
        return last_;
    }

    void advance(double volume) {
        update_core(volume);
    }

    inline const PercentageVolumeResult &last() const {
        return last_;
    }

private:
    inline void update_core(double volume) {
        ++counter_;

        const double ema_2 = oscillator_2_.update(volume);
        const double pvo = 100.0 * (oscillator_1_.update(volume) - ema_2) / ema_2;
        const double signal = signal_.update(pvo);
        const double histogram = pvo - signal;

        if (counter_ < 0) {
            last_ = {nan(), nan(), nan()};
        } else {
            last_ = {pvo, signal, histogram};
        }
    }

public:
    PercentageVolumeBatchResult batch(const InputArray &volume) {
        const std::size_t size = volume.shape(0);
        std::vector<double> pvos(size);
        std::vector<double> signals(size);
        std::vector<double> histograms(size);

        for (std::size_t i = 0; i < size; ++i) {
            ++counter_;
            const double ema_2 = oscillator_2_.update(volume(i));
            const double pvo = 100.0 * (oscillator_1_.update(volume(i)) - ema_2) / ema_2;
            const double signal = signal_.update(pvo);

            if (counter_ < 0) {
                pvos[i] = nan();
                signals[i] = nan();
                histograms[i] = nan();
            } else {
                pvos[i] = pvo;
                signals[i] = signal;
                histograms[i] = pvo - signal;
            }
        }

        return {make_array(std::move(pvos)), make_array(std::move(signals)), make_array(std::move(histograms))};
    }

private:
    EMA oscillator_1_;
    EMA oscillator_2_;
    EMA signal_;
    long counter_;
    PercentageVolumeResult last_;
};

class ROC {
public:
    explicit ROC(int window, bool fillna = true)
        : history_(static_cast<std::size_t>(std::max(window, 1)), 0.0),
          index_(0),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double close) {
        const double close_ago = update_history(close);
        if (!fillna_ && counter_ < window_) {
            ++counter_;
            return nan();
        }

        if (close_ago == 0.0) {
            ++counter_;
            return 0.0;
        }

        ++counter_;
        return ((close - close_ago) / close_ago) * 100.0;
    }

    nb::object batch(const InputArray &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        const double *values = close.data();

        for (std::size_t i = 0; i < size; ++i) {
            const double value = values[i];
            const double close_ago = update_history(value);
            if (!fillna_ && counter_ < window_) {
                output[i] = nan();
            } else if (close_ago == 0.0) {
                output[i] = 0.0;
            } else {
                output[i] = 100.0 * ((value - close_ago) / close_ago);
            }
            ++counter_;
        }

        return make_array(std::move(output));
    }

private:
    double update_history(double close) {
        const std::size_t index = static_cast<std::size_t>(index_);
        const double close_ago = history_[index];
        history_[index] = close;
        ++index_;
        if (index_ == static_cast<int>(history_.size())) {
            index_ = 0;
        }
        return close_ago;
    }

    std::vector<double> history_;
    int index_;
    int window_;
    bool fillna_;
    long counter_;
};

class RSI {
public:
    RSI(int window = 14, bool fillna = true)
        : fillna_(fillna),
          window_(window),
          counter_(0),
          prev_(0.0),
          high_(0.0),
          low_(0.0) {}

    double update(double value) {
        double retval = nan();

        if (counter_ == 0) {
            prev_ = value;
            retval = fillna_ ? 50.0 : nan();
        } else if (counter_ <= window_) {
            if (value < prev_) {
                low_ = (low_ * (counter_ - 1) + prev_ - value) / counter_;
            } else if (value > prev_) {
                high_ = (high_ * (counter_ - 1) + value - prev_) / counter_;
            }

            if (!fillna_) {
                retval = nan();
            } else if (low_ == 0.0) {
                retval = 100.0;
            } else {
                retval = 100.0 - (100.0 / (1.0 + (high_ / window_) / (low_ / window_)));
            }
        } else {
            if (value < prev_) {
                low_ = (low_ * (window_ - 1) + prev_ - value) / window_;
            } else if (low_ == 0.0) {
                retval = 100.0;
            } else if (value > prev_) {
                high_ = (high_ * (window_ - 1) + value - prev_) / window_;
            }

            if (std::isnan(retval)) {
                retval = 100.0 - (100.0 / (1.0 + (high_ / window_) / (low_ / window_)));
            }
        }

        prev_ = value;
        ++counter_;
        return retval;
    }

private:
    bool fillna_;
    int window_;
    long counter_;
    double prev_;
    double high_;
    double low_;
};

class ConnorsRSI {
public:
    ConnorsRSI(int rsi_window = 3, int streak_rsi_window = 2, int rank_window = 100, bool fillna = true)
        : price_rsi_(std::max(rsi_window, 1), fillna),
          streak_rsi_(std::max(streak_rsi_window, 1), fillna),
          changes_(std::max(rank_window, 1)),
          rank_window_(std::max(rank_window, 1)),
          previous_(0.0),
          streak_(0.0),
          first_(true),
          fillna_(fillna),
          last_(nan()) {}

    double update(double close) {
        const double rsi = price_rsi_.update(close);
        double change = 0.0;
        if (!first_) {
            change = close - previous_;
            if (change > 0.0) {
                streak_ = streak_ > 0.0 ? streak_ + 1.0 : 1.0;
            } else if (change < 0.0) {
                streak_ = streak_ < 0.0 ? streak_ - 1.0 : -1.0;
            } else {
                streak_ = 0.0;
            }
        }

        const double streak_rsi = streak_rsi_.update(streak_);
        changes_.push(change);
        std::size_t below = 0;
        for (std::size_t i = 0; i < changes_.size(); ++i) {
            if (changes_.at(i) < change) {
                ++below;
            }
        }
        const double percent_rank = 100.0 * static_cast<double>(below) / static_cast<double>(changes_.size());
        previous_ = close;
        first_ = false;

        last_ = (!fillna_ && !changes_.full()) ? nan() : (rsi + streak_rsi + percent_rank) / 3.0;
        return last_;
    }

    void advance(double close) {
        (void) update(close);
    }

    double last() const {
        return last_;
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(values[i]));
        }
        return make_array(std::move(output));
    }

private:
    RSI price_rsi_;
    RSI streak_rsi_;
    RollingBuffer changes_;
    int rank_window_;
    double previous_;
    double streak_;
    bool first_;
    bool fillna_;
    double last_;
};

class RelativeVigorIndex {
public:
    RelativeVigorIndex(int window = 10, bool fillna = true)
        : numerator_(std::max(window, 1)),
          denominator_(std::max(window, 1)),
          close_open_(4),
          high_low_(4),
          values_(4),
          numerator_sum_(0.0),
          denominator_sum_(0.0),
          window_(std::max(window, 1)),
          count_(0),
          fillna_(fillna),
          last_{nan(), nan()} {}

    RelativeVigorIndexResult update(double open, double high, double low, double close) {
        close_open_.push(close - open);
        high_low_.push(high - low);
        const double num = weighted_recent(close_open_);
        const double den = weighted_recent(high_low_);
        rolling_sum_push(numerator_, numerator_sum_, num);
        rolling_sum_push(denominator_, denominator_sum_, den);

        const double rvi = safe_divide(numerator_sum_, denominator_sum_);
        values_.push(rvi);
        const double signal = weighted_recent(values_);
        ++count_;
        if (!fillna_ && count_ < static_cast<std::size_t>(window_ + 3)) {
            last_ = {nan(), nan()};
        } else {
            last_ = {rvi, signal};
        }
        return last_;
    }

    void advance(double open, double high, double low, double close) {
        (void) update(open, high, low, close);
    }

    const RelativeVigorIndexResult &last() const {
        return last_;
    }

    template <typename Array0, typename Array1, typename Array2, typename Array3>
    RelativeVigorIndexBatchResult batch_array(const Array0 &open, const Array1 &high, const Array2 &low, const Array3 &close) {
        const std::size_t size = open.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        require_same_size(size, close.shape(0));
        std::vector<double> rvi(size);
        std::vector<double> signal(size);
        const auto *open_values = open.data();
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        const auto *close_values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const RelativeVigorIndexResult out = update(
                static_cast<double>(open_values[i]),
                static_cast<double>(high_values[i]),
                static_cast<double>(low_values[i]),
                static_cast<double>(close_values[i]));
            rvi[i] = out.rvi;
            signal[i] = out.signal;
        }
        return {make_array(std::move(rvi)), make_array(std::move(signal))};
    }

private:
    static double weighted_recent(const RollingBuffer &buffer) {
        static constexpr double weights[4] = {1.0, 2.0, 2.0, 1.0};
        double total = 0.0;
        double denom = 0.0;
        const std::size_t count = std::min<std::size_t>(buffer.size(), 4);
        for (std::size_t i = 0; i < count; ++i) {
            const double weight = weights[i];
            total += weight * buffer.at(buffer.size() - 1 - i);
            denom += weight;
        }
        return safe_divide(total, denom);
    }

    RollingBuffer numerator_;
    RollingBuffer denominator_;
    RollingBuffer close_open_;
    RollingBuffer high_low_;
    RollingBuffer values_;
    double numerator_sum_;
    double denominator_sum_;
    int window_;
    std::size_t count_;
    bool fillna_;
    RelativeVigorIndexResult last_;
};

class KlingerVolumeOscillator {
public:
    KlingerVolumeOscillator(int fast = 34, int slow = 55, int signal_window = 13, bool fillna = true)
        : fast_(std::max(fast, 1), true),
          slow_(std::max(slow, 1), true),
          signal_(std::max(signal_window, 1), true),
          previous_hlc_(0.0),
          previous_dm_(0.0),
          cumulative_measure_(0.0),
          previous_trend_(1.0),
          count_(0),
          warmup_(std::max({std::max(fast, 1), std::max(slow, 1), std::max(signal_window, 1)})),
          fillna_(fillna),
          last_{nan(), nan(), nan()} {}

    KlingerVolumeOscillatorResult update(double close, double high, double low, double volume) {
        const double hlc = high + low + close;
        const double dm = high - low;
        double trend = previous_trend_;
        if (count_ > 0) {
            trend = hlc > previous_hlc_ ? 1.0 : -1.0;
        }
        cumulative_measure_ = count_ == 0 ? dm : (trend == previous_trend_ ? cumulative_measure_ + dm : previous_dm_ + dm);
        const double temp = cumulative_measure_ == 0.0 ? 0.0 : std::abs(2.0 * (dm / cumulative_measure_ - 1.0));
        const double vf = volume * temp * trend * 100.0;
        const double kvo = fast_.update_always_filled(vf) - slow_.update_always_filled(vf);
        const double signal_value = signal_.update_always_filled(kvo);

        previous_hlc_ = hlc;
        previous_dm_ = dm;
        previous_trend_ = trend;
        ++count_;
        if (!fillna_ && count_ < static_cast<std::size_t>(warmup_)) {
            last_ = {nan(), nan(), nan()};
        } else {
            last_ = {kvo, signal_value, kvo - signal_value};
        }
        return last_;
    }

    void advance(double close, double high, double low, double volume) {
        (void) update(close, high, low, volume);
    }

    const KlingerVolumeOscillatorResult &last() const {
        return last_;
    }

    template <typename Array0, typename Array1, typename Array2, typename Array3>
    KlingerVolumeOscillatorBatchResult batch_array(const Array0 &close, const Array1 &high, const Array2 &low, const Array3 &volume) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        require_same_size(size, volume.shape(0));
        std::vector<double> kvo(size);
        std::vector<double> signal(size);
        std::vector<double> histogram(size);
        const auto *close_values = close.data();
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        const auto *volume_values = volume.data();
        for (std::size_t i = 0; i < size; ++i) {
            const KlingerVolumeOscillatorResult out = update(
                static_cast<double>(close_values[i]),
                static_cast<double>(high_values[i]),
                static_cast<double>(low_values[i]),
                static_cast<double>(volume_values[i]));
            kvo[i] = out.kvo;
            signal[i] = out.signal;
            histogram[i] = out.histogram;
        }
        return {
            make_array(std::move(kvo)),
            make_array(std::move(signal)),
            make_array(std::move(histogram)),
        };
    }

private:
    EMA fast_;
    EMA slow_;
    EMA signal_;
    double previous_hlc_;
    double previous_dm_;
    double cumulative_measure_;
    double previous_trend_;
    std::size_t count_;
    int warmup_;
    bool fillna_;
    KlingerVolumeOscillatorResult last_;
};

class ElderRayIndex {
public:
    ElderRayIndex(int window = 13, bool fillna = true)
        : ema_(std::max(window, 1), fillna),
          last_{nan(), nan()} {}

    ElderRayIndexResult update(double close, double high, double low) {
        const double value = ema_.update(close);
        last_ = {high - value, low - value};
        return last_;
    }

    void advance(double close, double high, double low) {
        (void) update(close, high, low);
    }

    const ElderRayIndexResult &last() const {
        return last_;
    }

    template <typename Array0, typename Array1, typename Array2>
    ElderRayIndexBatchResult batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> bull(size);
        std::vector<double> bear(size);
        const auto *close_values = close.data();
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        for (std::size_t i = 0; i < size; ++i) {
            const ElderRayIndexResult out = update(
                static_cast<double>(close_values[i]),
                static_cast<double>(high_values[i]),
                static_cast<double>(low_values[i]));
            bull[i] = out.bull_power;
            bear[i] = out.bear_power;
        }
        return {make_array(std::move(bull)), make_array(std::move(bear))};
    }

private:
    EMA ema_;
    ElderRayIndexResult last_;
};

class CoppockCurve {
public:
    CoppockCurve(int wma_window = 10, int long_roc = 14, int short_roc = 11, bool fillna = true)
        : long_delay_(std::max(long_roc, 1), fillna),
          short_delay_(std::max(short_roc, 1), fillna),
          wma_(std::max(wma_window, 1), fillna),
          warmup_(std::max(long_roc, short_roc) + std::max(wma_window, 1) - 1),
          count_(0),
          fillna_(fillna),
          last_(nan()) {}

    double update(double close) {
        const double long_previous = long_delay_.update(close);
        const double short_previous = short_delay_.update(close);
        const double value = safe_divide(close - long_previous, long_previous) * 100.0 +
                             safe_divide(close - short_previous, short_previous) * 100.0;
        last_ = wma_.update(value);
        ++count_;
        return (!fillna_ && count_ < warmup_) ? nan() : last_;
    }

    void advance(double close) {
        (void) update(close);
    }

    double last() const {
        return last_;
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(values[i]));
        }
        return make_array(std::move(output));
    }

private:
    Delay long_delay_;
    Delay short_delay_;
    WeightedMovingAverage wma_;
    int warmup_;
    int count_;
    bool fillna_;
    double last_;
};

class FisherTransform {
public:
    FisherTransform(int window = 10, bool fillna = true)
        : highs_(std::max(window, 1), true),
          lows_(std::max(window, 1), false),
          value_(0.0),
          fisher_(0.0),
          fillna_(fillna),
          last_(nan()) {}

    double update(double high, double low) {
        highs_.push(high);
        lows_.push(low);
        const double price = (high + low) * 0.5;
        const double highest = highs_.value();
        const double lowest = lows_.value();
        double normalized = safe_divide(2.0 * (price - lowest), highest - lowest) - 1.0;
        normalized = std::clamp(normalized, -0.999, 0.999);
        value_ = std::clamp(0.33 * normalized + 0.67 * value_, -0.999, 0.999);
        fisher_ = 0.5 * std::log((1.0 + value_) / (1.0 - value_)) + 0.5 * fisher_;
        last_ = (!fillna_ && !highs_.full()) ? nan() : fisher_;
        return last_;
    }

    void advance(double high, double low) {
        (void) update(high, low);
    }

    double last() const {
        return last_;
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &high, const Array1 &low) {
        const std::size_t size = high.shape(0);
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(high_values[i]), static_cast<double>(low_values[i]));
        }
        return make_array(std::move(output));
    }

private:
    RollingExtreme highs_;
    RollingExtreme lows_;
    double value_;
    double fisher_;
    bool fillna_;
    double last_;
};

class FractalAdaptiveMovingAverage {
public:
    FractalAdaptiveMovingAverage(int window = 16, bool fillna = true)
        : window_(std::max(window, 2)),
          half_(std::max(window_, 2) / 2),
          values_(std::max(window_, 2)),
          first_(true),
          value_(nan()),
          fillna_(fillna) {}

    double update(double close) {
        values_.push(close);
        if (first_) {
            value_ = close;
            first_ = false;
        }

        if (!fillna_ && !values_.full()) {
            return nan();
        }

        double alpha = 1.0;
        if (values_.size() >= static_cast<std::size_t>(window_)) {
            const auto first_half = range(0, static_cast<std::size_t>(half_));
            const auto second_half = range(static_cast<std::size_t>(half_), static_cast<std::size_t>(window_ - half_));
            const auto full = range(0, static_cast<std::size_t>(window_));
            const double n1 = (first_half.second - first_half.first) / static_cast<double>(half_);
            const double n2 = (second_half.second - second_half.first) / static_cast<double>(window_ - half_);
            const double n3 = (full.second - full.first) / static_cast<double>(window_);
            if (n1 > 0.0 && n2 > 0.0 && n3 > 0.0) {
                const double dimension = (std::log(n1 + n2) - std::log(n3)) / std::log(2.0);
                alpha = std::clamp(std::exp(-4.6 * (dimension - 1.0)), 0.01, 1.0);
            }
        }

        value_ = alpha * close + (1.0 - alpha) * value_;
        return value_;
    }

    void advance(double close) {
        (void) update(close);
    }

    double last() const {
        return value_;
    }

    template <typename Array>
    nb::object batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(values[i]));
        }
        return make_array(std::move(output));
    }

private:
    std::pair<double, double> range(std::size_t offset, std::size_t count) const {
        const std::size_t available = values_.size();
        const std::size_t start = available - static_cast<std::size_t>(window_) + offset;
        double lowest = values_.at(start);
        double highest = lowest;
        for (std::size_t i = 1; i < count; ++i) {
            const double value = values_.at(start + i);
            lowest = std::min(lowest, value);
            highest = std::max(highest, value);
        }
        return {lowest, highest};
    }

    int window_;
    int half_;
    RollingBuffer values_;
    bool first_;
    double value_;
    bool fillna_;
};

class MesaAdaptiveMovingAverage {
public:
    MesaAdaptiveMovingAverage(double fast_limit = 0.5, double slow_limit = 0.05, bool fillna = true)
        : fast_limit_(std::clamp(fast_limit, 0.01, 1.0)),
          slow_limit_(std::clamp(slow_limit, 0.001, fast_limit_)),
          prices_(7),
          smooth_(7),
          detrender_(7),
          q1_(7),
          i1_(7),
          period_(6.0),
          smooth_period_(6.0),
          prev_i2_(0.0),
          prev_q2_(0.0),
          re_(0.0),
          im_(0.0),
          prev_phase_(0.0),
          mama_(nan()),
          fama_(nan()),
          count_(0),
          fillna_(fillna),
          last_{nan(), nan()} {}

    MesaAdaptiveMovingAverageResult update(double price) {
        prices_.push(price);
        const double smoothed = (4.0 * recent(prices_, 0) + 3.0 * recent(prices_, 1) + 2.0 * recent(prices_, 2) + recent(prices_, 3)) / 10.0;
        smooth_.push(smoothed);

        const double adjust = 0.075 * period_ + 0.54;
        const double detrender = hilbert(smooth_, adjust);
        detrender_.push(detrender);
        const double q1 = hilbert(detrender_, adjust);
        q1_.push(q1);
        const double i1 = recent(detrender_, 3);
        i1_.push(i1);
        const double ji = hilbert(i1_, adjust);
        const double jq = hilbert(q1_, adjust);

        double i2 = i1 - jq;
        double q2 = q1 + ji;
        i2 = 0.2 * i2 + 0.8 * prev_i2_;
        q2 = 0.2 * q2 + 0.8 * prev_q2_;
        re_ = 0.2 * (i2 * prev_i2_ + q2 * prev_q2_) + 0.8 * re_;
        im_ = 0.2 * (i2 * prev_q2_ - q2 * prev_i2_) + 0.8 * im_;

        if (im_ != 0.0 && re_ != 0.0) {
            double period = 360.0 / (std::atan(im_ / re_) * 180.0 / std::acos(-1.0));
            period = std::clamp(period, 0.67 * period_, 1.5 * period_);
            period_ = std::clamp(period, 6.0, 50.0);
        }
        smooth_period_ = 0.33 * period_ + 0.67 * smooth_period_;

        const double phase = i1 != 0.0 ? std::atan(q1 / i1) * 180.0 / std::acos(-1.0) : prev_phase_;
        double delta_phase = prev_phase_ - phase;
        if (delta_phase < 1.0) {
            delta_phase = 1.0;
        }
        const double alpha = std::max(fast_limit_ / delta_phase, slow_limit_);

        if (count_ == 0) {
            mama_ = price;
            fama_ = price;
        } else {
            mama_ = alpha * price + (1.0 - alpha) * mama_;
            fama_ = 0.5 * alpha * mama_ + (1.0 - 0.5 * alpha) * fama_;
        }

        prev_i2_ = i2;
        prev_q2_ = q2;
        prev_phase_ = phase;
        ++count_;
        last_ = (!fillna_ && count_ < 7) ? MesaAdaptiveMovingAverageResult{nan(), nan()} : MesaAdaptiveMovingAverageResult{mama_, fama_};
        return last_;
    }

    void advance(double price) {
        (void) update(price);
    }

    const MesaAdaptiveMovingAverageResult &last() const {
        return last_;
    }

    template <typename Array>
    MesaAdaptiveMovingAverageBatchResult batch_array(const Array &price) {
        const std::size_t size = price.shape(0);
        std::vector<double> mama(size);
        std::vector<double> fama(size);
        const auto *values = price.data();
        for (std::size_t i = 0; i < size; ++i) {
            const MesaAdaptiveMovingAverageResult out = update(static_cast<double>(values[i]));
            mama[i] = out.mama;
            fama[i] = out.fama;
        }
        return {make_array(std::move(mama)), make_array(std::move(fama))};
    }

private:
    static double recent(const RollingBuffer &buffer, std::size_t offset) {
        if (offset >= buffer.size()) {
            return 0.0;
        }
        return buffer.at(buffer.size() - 1 - offset);
    }

    static double hilbert(const RollingBuffer &buffer, double adjust) {
        return (0.0962 * recent(buffer, 0) + 0.5769 * recent(buffer, 2) -
                0.5769 * recent(buffer, 4) - 0.0962 * recent(buffer, 6)) * adjust;
    }

    double fast_limit_;
    double slow_limit_;
    RollingBuffer prices_;
    RollingBuffer smooth_;
    RollingBuffer detrender_;
    RollingBuffer q1_;
    RollingBuffer i1_;
    double period_;
    double smooth_period_;
    double prev_i2_;
    double prev_q2_;
    double re_;
    double im_;
    double prev_phase_;
    double mama_;
    double fama_;
    std::size_t count_;
    bool fillna_;
    MesaAdaptiveMovingAverageResult last_;
};

class EhlersOptimalTrackingFilter {
public:
    explicit EhlersOptimalTrackingFilter(bool fillna = true)
        : previous_price_(0.0),
          value1_(0.0),
          value2_(0.0),
          value3_(nan()),
          count_(0),
          fillna_(fillna) {}

    double update(double high, double low) {
        const double price = (high + low) * 0.5;
        if (count_ == 0) {
            previous_price_ = price;
            value3_ = price;
        }
        value1_ = 0.2 * (price - previous_price_) + 0.8 * value1_;
        value2_ = 0.1 * (high - low) + 0.8 * value2_;
        if (count_ > 5 && value2_ != 0.0) {
            const double lambda = std::abs(value1_ / value2_);
            const double lambda2 = lambda * lambda;
            const double alpha = (-lambda2 + std::sqrt(lambda2 * lambda2 + 16.0 * lambda2)) / 8.0;
            value3_ = alpha * price + (1.0 - alpha) * value3_;
        } else {
            value3_ = price;
        }
        previous_price_ = price;
        ++count_;
        return (!fillna_ && count_ <= 5) ? nan() : value3_;
    }

    void advance(double high, double low) {
        (void) update(high, low);
    }

    double last() const {
        return value3_;
    }

    template <typename Array0, typename Array1>
    nb::object batch_array(const Array0 &high, const Array1 &low) {
        const std::size_t size = high.shape(0);
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(static_cast<double>(high_values[i]), static_cast<double>(low_values[i]));
        }
        return make_array(std::move(output));
    }

private:
    double previous_price_;
    double value1_;
    double value2_;
    double value3_;
    std::size_t count_;
    bool fillna_;
};

class KalmanHedgeRatio {
public:
    KalmanHedgeRatio(double initial_hedge_ratio = 1.0,
                     double initial_intercept = 0.0,
                     double initial_variance = 1.0,
                     double process_variance = 1.0e-5,
                     double measurement_variance = 1.0,
                     bool fillna = true)
        : initial_hedge_ratio_(initial_hedge_ratio),
          initial_intercept_(initial_intercept),
          initial_variance_(variance_floor(initial_variance)),
          process_variance_(variance_floor(process_variance)),
          measurement_variance_(variance_floor(measurement_variance)),
          initialized_(false),
          count_(0),
          fillna_(fillna),
          last_{nan(), nan(), nan()} {}

    KalmanHedgeRatioResult update(double x, double y) {
        ensure_initialized();
        filter_.H = kalman::Mat<1, 2>{x, 1.0};
        (void) filter_.update(kalman::Vec<1>{y});
        const double hedge_ratio = filter_.x[0];
        const double intercept = filter_.x[1];
        const double spread = y - (hedge_ratio * x + intercept);
        ++count_;
        last_ = (!fillna_ && count_ < 2) ? KalmanHedgeRatioResult{nan(), nan(), nan()} : KalmanHedgeRatioResult{hedge_ratio, intercept, spread};
        return last_;
    }

    void advance(double x, double y) {
        (void) update(x, y);
    }

    const KalmanHedgeRatioResult &last() const {
        return last_;
    }

    template <typename Array0, typename Array1>
    KalmanHedgeRatioBatchResult batch_array(const Array0 &x, const Array1 &y) {
        const std::size_t size = x.shape(0);
        require_same_size(size, y.shape(0));
        std::vector<double> hedge(size);
        std::vector<double> intercept(size);
        std::vector<double> spread(size);
        const auto *x_values = x.data();
        const auto *y_values = y.data();
        for (std::size_t i = 0; i < size; ++i) {
            const KalmanHedgeRatioResult out = update(static_cast<double>(x_values[i]), static_cast<double>(y_values[i]));
            hedge[i] = out.hedge_ratio;
            intercept[i] = out.intercept;
            spread[i] = out.spread;
        }
        return {make_array(std::move(hedge)), make_array(std::move(intercept)), make_array(std::move(spread))};
    }

private:
    static double variance_floor(double value, double minimum = kalman::kDefaultVarianceFloor) {
        return (!std::isfinite(value) || value < minimum) ? minimum : value;
    }

    void ensure_initialized() {
        if (initialized_) {
            return;
        }
        filter_.x = kalman::Vec<2>{initial_hedge_ratio_, initial_intercept_};
        filter_.P = kalman::Mat<2, 2>{initial_variance_, 0.0, 0.0, initial_variance_};
        filter_.F = kalman::Mat<2, 2>::identity();
        filter_.Q = kalman::Mat<2, 2>{process_variance_, 0.0, 0.0, process_variance_};
        filter_.R = kalman::Mat<1, 1>{measurement_variance_};
        initialized_ = true;
    }

    double initial_hedge_ratio_;
    double initial_intercept_;
    double initial_variance_;
    double process_variance_;
    double measurement_variance_;
    kalman::LinearKalmanFilter<2, 1> filter_;
    bool initialized_;
    std::size_t count_;
    bool fillna_;
    KalmanHedgeRatioResult last_;
};

class KalmanRegressionChannel {
public:
    KalmanRegressionChannel(double initial_slope = 1.0,
                            double initial_intercept = 0.0,
                            double initial_variance = 1.0,
                            double process_variance = 1.0e-5,
                            double measurement_variance = 1.0,
                            double multiplier = 2.0,
                            bool fillna = true)
        : initial_slope_(initial_slope),
          initial_intercept_(initial_intercept),
          initial_variance_(variance_floor(initial_variance)),
          process_variance_(variance_floor(process_variance)),
          measurement_variance_(variance_floor(measurement_variance)),
          multiplier_(std::isfinite(multiplier) ? std::abs(multiplier) : 2.0),
          initialized_(false),
          count_(0),
          fillna_(fillna),
          last_{nan(), nan(), nan(), nan(), nan(), nan()} {}

    KalmanRegressionChannelResult update(double x, double y) {
        ensure_initialized();
        filter_.H = kalman::Mat<1, 2>{x, 1.0};
        const auto stats = filter_.update(kalman::Vec<1>{y});
        const double slope = filter_.x[0];
        const double intercept = filter_.x[1];
        const double middle = slope * x + intercept;
        const double spread = y - middle;
        const double band = multiplier_ * std::sqrt(std::max(stats.S(0, 0), 0.0));
        ++count_;
        last_ = (!fillna_ && count_ < 2) ? KalmanRegressionChannelResult{nan(), nan(), nan(), nan(), nan(), nan()} :
            KalmanRegressionChannelResult{slope, intercept, middle, middle + band, middle - band, spread};
        return last_;
    }

    void advance(double x, double y) {
        (void) update(x, y);
    }

    const KalmanRegressionChannelResult &last() const {
        return last_;
    }

    template <typename Array0, typename Array1>
    KalmanRegressionChannelBatchResult batch_array(const Array0 &x, const Array1 &y) {
        const std::size_t size = x.shape(0);
        require_same_size(size, y.shape(0));
        std::vector<double> slope(size);
        std::vector<double> intercept(size);
        std::vector<double> middle(size);
        std::vector<double> upper(size);
        std::vector<double> lower(size);
        std::vector<double> spread(size);
        const auto *x_values = x.data();
        const auto *y_values = y.data();
        for (std::size_t i = 0; i < size; ++i) {
            const KalmanRegressionChannelResult out = update(static_cast<double>(x_values[i]), static_cast<double>(y_values[i]));
            slope[i] = out.slope;
            intercept[i] = out.intercept;
            middle[i] = out.middle;
            upper[i] = out.upper;
            lower[i] = out.lower;
            spread[i] = out.spread;
        }
        return {
            make_array(std::move(slope)),
            make_array(std::move(intercept)),
            make_array(std::move(middle)),
            make_array(std::move(upper)),
            make_array(std::move(lower)),
            make_array(std::move(spread)),
        };
    }

private:
    static double variance_floor(double value, double minimum = kalman::kDefaultVarianceFloor) {
        return (!std::isfinite(value) || value < minimum) ? minimum : value;
    }

    void ensure_initialized() {
        if (initialized_) {
            return;
        }
        filter_.x = kalman::Vec<2>{initial_slope_, initial_intercept_};
        filter_.P = kalman::Mat<2, 2>{initial_variance_, 0.0, 0.0, initial_variance_};
        filter_.F = kalman::Mat<2, 2>::identity();
        filter_.Q = kalman::Mat<2, 2>{process_variance_, 0.0, 0.0, process_variance_};
        filter_.R = kalman::Mat<1, 1>{measurement_variance_};
        initialized_ = true;
    }

    double initial_slope_;
    double initial_intercept_;
    double initial_variance_;
    double process_variance_;
    double measurement_variance_;
    double multiplier_;
    kalman::LinearKalmanFilter<2, 1> filter_;
    bool initialized_;
    std::size_t count_;
    bool fillna_;
    KalmanRegressionChannelResult last_;
};

class TwoFactorKalmanTrendFilter {
public:
    TwoFactorKalmanTrendFilter(double short_persistence = 0.55,
                               double short_to_long = 0.45,
                               double long_persistence = 0.98,
                               double short_loading = 1.0,
                               double long_loading = 1.0,
                               double process_short_variance = 1.0e-3,
                               double process_long_variance = 1.0e-5,
                               double measurement_variance = 0.25,
                               bool fillna = true)
        : short_persistence_(short_persistence),
          short_to_long_(short_to_long),
          long_persistence_(long_persistence),
          short_loading_(short_loading),
          long_loading_(long_loading),
          process_short_variance_(variance_floor(process_short_variance)),
          process_long_variance_(variance_floor(process_long_variance)),
          measurement_variance_(variance_floor(measurement_variance)),
          initialized_(false),
          count_(0),
          fillna_(fillna),
          last_{nan(), nan(), nan()} {}

    TwoFactorKalmanTrendFilterResult update(double close) {
        if (!initialized_) {
            initialize(close);
        }
        (void) filter_.update(kalman::Vec<1>{close});
        ++count_;
        const double value = short_loading_ * filter_.x[0] + long_loading_ * filter_.x[1];
        last_ = (!fillna_ && count_ < 2) ? TwoFactorKalmanTrendFilterResult{nan(), nan(), nan()} :
            TwoFactorKalmanTrendFilterResult{filter_.x[0], filter_.x[1], value};
        return last_;
    }

    void advance(double close) {
        (void) update(close);
    }

    const TwoFactorKalmanTrendFilterResult &last() const {
        return last_;
    }

    template <typename Array>
    TwoFactorKalmanTrendFilterBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> short_trend(size);
        std::vector<double> long_trend(size);
        std::vector<double> value(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const TwoFactorKalmanTrendFilterResult out = update(static_cast<double>(values[i]));
            short_trend[i] = out.short_trend;
            long_trend[i] = out.long_trend;
            value[i] = out.value;
        }
        return {make_array(std::move(short_trend)), make_array(std::move(long_trend)), make_array(std::move(value))};
    }

private:
    static double variance_floor(double value, double minimum = kalman::kDefaultVarianceFloor) {
        return (!std::isfinite(value) || value < minimum) ? minimum : value;
    }

    void initialize(double close) {
        const double denom = std::abs(short_loading_) + std::abs(long_loading_);
        const double seed = denom == 0.0 ? close : close / denom;
        filter_.x = kalman::Vec<2>{seed, seed};
        filter_.P = kalman::Mat<2, 2>{1.0, 0.0, 0.0, 1.0};
        filter_.F = kalman::Mat<2, 2>{short_persistence_, short_to_long_, 0.0, long_persistence_};
        filter_.Q = kalman::Mat<2, 2>{process_short_variance_, 0.0, 0.0, process_long_variance_};
        filter_.H = kalman::Mat<1, 2>{short_loading_, long_loading_};
        filter_.R = kalman::Mat<1, 1>{measurement_variance_};
        initialized_ = true;
    }

    double short_persistence_;
    double short_to_long_;
    double long_persistence_;
    double short_loading_;
    double long_loading_;
    double process_short_variance_;
    double process_long_variance_;
    double measurement_variance_;
    kalman::LinearKalmanFilter<2, 1> filter_;
    bool initialized_;
    std::size_t count_;
    bool fillna_;
    TwoFactorKalmanTrendFilterResult last_;
};

class KalmanExtremumTrend {
public:
    KalmanExtremumTrend(int window = 14,
                        double initial_price = nan(),
                        double initial_velocity = 0.0,
                        double dt = 1.0,
                        double position_variance = 1.0,
                        double velocity_variance = 1.0,
                        double process_position_variance = 1.0e-4,
                        double process_velocity_variance = 1.0e-3,
                        double measurement_variance = 0.25,
                        bool fillna = true)
        : trend_(initial_price,
                 initial_velocity,
                 dt,
                 position_variance,
                 velocity_variance,
                 process_position_variance,
                 process_velocity_variance,
                 measurement_variance,
                 fillna),
          highs_(std::max(window, 1), true),
          lows_(std::max(window, 1), false),
          previous_trend_(nan()),
          count_(0),
          fillna_(fillna),
          last_{nan(), nan(), nan()} {}

    KalmanExtremumTrendResult update(double close, double high, double low) {
        const double trend = trend_.update(close);
        highs_.push(high);
        lows_.push(low);
        const double oscillator = safe_divide(100.0 * (close - lows_.value()), highs_.value() - lows_.value(), 50.0);
        double signal = 0.0;
        if (std::isfinite(previous_trend_)) {
            if (trend > previous_trend_ && oscillator >= 50.0) {
                signal = 1.0;
            } else if (trend < previous_trend_ && oscillator < 50.0) {
                signal = -1.0;
            }
        }
        previous_trend_ = trend;
        ++count_;
        last_ = (!fillna_ && !highs_.full()) ? KalmanExtremumTrendResult{nan(), nan(), nan()} :
            KalmanExtremumTrendResult{trend, oscillator, signal};
        return last_;
    }

    void advance(double close, double high, double low) {
        (void) update(close, high, low);
    }

    const KalmanExtremumTrendResult &last() const {
        return last_;
    }

    template <typename Array0, typename Array1, typename Array2>
    KalmanExtremumTrendBatchResult batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> trend(size);
        std::vector<double> oscillator(size);
        std::vector<double> signal(size);
        const auto *close_values = close.data();
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        for (std::size_t i = 0; i < size; ++i) {
            const KalmanExtremumTrendResult out = update(
                static_cast<double>(close_values[i]),
                static_cast<double>(high_values[i]),
                static_cast<double>(low_values[i]));
            trend[i] = out.trend;
            oscillator[i] = out.oscillator;
            signal[i] = out.signal;
        }
        return {make_array(std::move(trend)), make_array(std::move(oscillator)), make_array(std::move(signal))};
    }

private:
    KalmanMovingAverage trend_;
    RollingExtreme highs_;
    RollingExtreme lows_;
    double previous_trend_;
    std::size_t count_;
    bool fillna_;
    KalmanExtremumTrendResult last_;
};

class AlphaBetaGammaTrackingFilter {
public:
    AlphaBetaGammaTrackingFilter(double alpha = 0.65,
                                 double beta = 0.25,
                                 double gamma = 0.05,
                                 double dt = 1.0,
                                 double initial_price = nan(),
                                 double initial_velocity = 0.0,
                                 double initial_acceleration = 0.0,
                                 bool fillna = true)
        : alpha_(clamp_gain(alpha)),
          beta_(clamp_gain(beta)),
          gamma_(clamp_gain(gamma)),
          dt_(dt > 0.0 ? dt : 1.0),
          initial_price_(initial_price),
          price_(0.0),
          velocity_(std::isfinite(initial_velocity) ? initial_velocity : 0.0),
          acceleration_(std::isfinite(initial_acceleration) ? initial_acceleration : 0.0),
          initialized_(false),
          count_(0),
          fillna_(fillna),
          last_{nan(), nan(), nan(), nan()} {}

    AlphaBetaGammaTrackingFilterResult update(double close) {
        if (!initialized_) {
            initialized_ = true;
            if (!std::isfinite(initial_price_)) {
                price_ = close;
                last_ = AlphaBetaGammaTrackingFilterResult{price_, velocity_, acceleration_, 0.0};
                ++count_;
                return output_for_warmup();
            }
            price_ = initial_price_;
        }

        const double predicted_price = price_ + velocity_ * dt_ + 0.5 * acceleration_ * dt_ * dt_;
        const double predicted_velocity = velocity_ + acceleration_ * dt_;
        const double predicted_acceleration = acceleration_;
        const double residual = close - predicted_price;

        price_ = predicted_price + alpha_ * residual;
        velocity_ = predicted_velocity + beta_ * residual / dt_;
        acceleration_ = predicted_acceleration + 2.0 * gamma_ * residual / (dt_ * dt_);
        last_ = AlphaBetaGammaTrackingFilterResult{price_, velocity_, acceleration_, residual};
        ++count_;
        return output_for_warmup();
    }

    void advance(double close) {
        (void) update(close);
    }

    const AlphaBetaGammaTrackingFilterResult &last() const {
        return last_;
    }

    template <typename Array>
    AlphaBetaGammaTrackingFilterBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> price(size);
        std::vector<double> velocity(size);
        std::vector<double> acceleration(size);
        std::vector<double> residual(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const AlphaBetaGammaTrackingFilterResult out = update(static_cast<double>(values[i]));
            price[i] = out.price;
            velocity[i] = out.velocity;
            acceleration[i] = out.acceleration;
            residual[i] = out.residual;
        }
        return {
            make_array(std::move(price)),
            make_array(std::move(velocity)),
            make_array(std::move(acceleration)),
            make_array(std::move(residual)),
        };
    }

private:
    static double clamp_gain(double value) {
        if (!std::isfinite(value)) {
            return 0.0;
        }
        return std::clamp(value, 0.0, 1.0);
    }

    AlphaBetaGammaTrackingFilterResult output_for_warmup() const {
        if (!fillna_ && count_ < 2) {
            return AlphaBetaGammaTrackingFilterResult{nan(), nan(), nan(), nan()};
        }
        return last_;
    }

    double alpha_;
    double beta_;
    double gamma_;
    double dt_;
    double initial_price_;
    double price_;
    double velocity_;
    double acceleration_;
    bool initialized_;
    std::size_t count_;
    bool fillna_;
    AlphaBetaGammaTrackingFilterResult last_;
};

class InteractingMultipleModelFilter {
public:
    InteractingMultipleModelFilter(double initial_price = nan(),
                                   double initial_velocity = 0.0,
                                   double dt = 1.0,
                                   double low_vol_process_variance = 1.0e-5,
                                   double high_vol_process_variance = 1.0e-2,
                                   double trend_process_variance = 1.0e-4,
                                   double chop_process_variance = 1.0e-3,
                                   double measurement_variance = 0.25,
                                   double stickiness = 0.96,
                                   bool fillna = true)
        : initial_price_(initial_price),
          initial_velocity_(std::isfinite(initial_velocity) ? initial_velocity : 0.0),
          dt_(dt > 0.0 ? dt : 1.0),
          stickiness_(std::clamp(std::isfinite(stickiness) ? stickiness : 0.96, 0.25, 0.999)),
          initialized_(false),
          count_(0),
          fillna_(fillna),
          last_{nan(), nan(), nan(), nan(), nan(), nan()} {
        const double measurement = variance_floor(measurement_variance);
        specs_[0] = ModelSpec{1.0, 0.85, variance_floor(low_vol_process_variance), variance_floor(low_vol_process_variance), variance_floor(0.5 * measurement)};
        specs_[1] = ModelSpec{1.0, 0.85, variance_floor(high_vol_process_variance), variance_floor(high_vol_process_variance), variance_floor(2.0 * measurement)};
        specs_[2] = ModelSpec{1.0, 1.0, variance_floor(0.25 * trend_process_variance), variance_floor(trend_process_variance), measurement};
        specs_[3] = ModelSpec{0.15, 0.20, variance_floor(chop_process_variance), variance_floor(2.0 * chop_process_variance), measurement};
    }

    InteractingMultipleModelFilterResult update(double close) {
        if (!initialized_) {
            initialize(std::isfinite(initial_price_) ? initial_price_ : close);
            if (!std::isfinite(initial_price_)) {
                ++count_;
                return output_for_warmup();
            }
        }

        std::array<ModelState, model_count_> mixed_models{};
        std::array<double, model_count_> predicted_probabilities{};
        mix_models(mixed_models, predicted_probabilities);

        std::array<double, model_count_> log_likelihoods{};
        double max_log_likelihood = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < model_count_; ++i) {
            const double log_likelihood = update_model(mixed_models[i], specs_[i], close);
            log_likelihoods[i] = log_likelihood;
            max_log_likelihood = std::max(max_log_likelihood, log_likelihood);
        }
        models_ = mixed_models;
        update_probabilities(predicted_probabilities, log_likelihoods, max_log_likelihood);
        update_combined_output();
        ++count_;
        return output_for_warmup();
    }

    void advance(double close) {
        (void) update(close);
    }

    const InteractingMultipleModelFilterResult &last() const {
        return last_;
    }

    template <typename Array>
    InteractingMultipleModelFilterBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> value(size);
        std::vector<double> velocity(size);
        std::vector<double> low_vol_probability(size);
        std::vector<double> high_vol_probability(size);
        std::vector<double> trend_probability(size);
        std::vector<double> chop_probability(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const InteractingMultipleModelFilterResult out = update(static_cast<double>(values[i]));
            value[i] = out.value;
            velocity[i] = out.velocity;
            low_vol_probability[i] = out.low_vol_probability;
            high_vol_probability[i] = out.high_vol_probability;
            trend_probability[i] = out.trend_probability;
            chop_probability[i] = out.chop_probability;
        }
        return {
            make_array(std::move(value)),
            make_array(std::move(velocity)),
            make_array(std::move(low_vol_probability)),
            make_array(std::move(high_vol_probability)),
            make_array(std::move(trend_probability)),
            make_array(std::move(chop_probability)),
        };
    }

private:
    struct ModelSpec {
        double position_velocity_loading;
        double velocity_persistence;
        double process_position_variance;
        double process_velocity_variance;
        double measurement_variance;
    };

    struct ModelState {
        double price;
        double velocity;
        double p00;
        double p01;
        double p11;
    };

    static constexpr std::size_t model_count_ = 4;
    static constexpr double pi_ = 3.141592653589793238462643383279502884;

    static double variance_floor(double value, double minimum = 1.0e-12) {
        return (!std::isfinite(value) || value < minimum) ? minimum : value;
    }

    double transition_probability(std::size_t from, std::size_t to) const {
        if (from == to) {
            return stickiness_;
        }
        return (1.0 - stickiness_) / static_cast<double>(model_count_ - 1);
    }

    void initialize(double close) {
        for (std::size_t i = 0; i < model_count_; ++i) {
            models_[i] = ModelState{close, initial_velocity_, 1.0, 0.0, 1.0};
            probabilities_[i] = 1.0 / static_cast<double>(model_count_);
        }
        initialized_ = true;
        update_combined_output();
    }

    void mix_models(std::array<ModelState, model_count_> &mixed_models, std::array<double, model_count_> &predicted_probabilities) const {
        for (std::size_t to = 0; to < model_count_; ++to) {
            double predicted_probability = 0.0;
            for (std::size_t from = 0; from < model_count_; ++from) {
                predicted_probability += probabilities_[from] * transition_probability(from, to);
            }
            predicted_probability = variance_floor(predicted_probability);
            predicted_probabilities[to] = predicted_probability;

            double price = 0.0;
            double velocity = 0.0;
            for (std::size_t from = 0; from < model_count_; ++from) {
                const double weight = probabilities_[from] * transition_probability(from, to) / predicted_probability;
                price += weight * models_[from].price;
                velocity += weight * models_[from].velocity;
            }

            double p00 = 0.0;
            double p01 = 0.0;
            double p11 = 0.0;
            for (std::size_t from = 0; from < model_count_; ++from) {
                const double weight = probabilities_[from] * transition_probability(from, to) / predicted_probability;
                const double dp = models_[from].price - price;
                const double dv = models_[from].velocity - velocity;
                p00 += weight * (models_[from].p00 + dp * dp);
                p01 += weight * (models_[from].p01 + dp * dv);
                p11 += weight * (models_[from].p11 + dv * dv);
            }
            mixed_models[to] = ModelState{price, velocity, variance_floor(p00), p01, variance_floor(p11)};
        }
    }

    double update_model(ModelState &model, const ModelSpec &spec, double close) const {
        const double loading = spec.position_velocity_loading * dt_;
        const double persistence = spec.velocity_persistence;
        const double predicted_price = model.price + loading * model.velocity;
        const double predicted_velocity = persistence * model.velocity;
        const double p00 = model.p00 + loading * model.p01 + loading * model.p01 + loading * loading * model.p11 + spec.process_position_variance;
        const double p01 = persistence * (model.p01 + loading * model.p11);
        const double p11 = persistence * persistence * model.p11 + spec.process_velocity_variance;
        const double innovation = close - predicted_price;
        const double innovation_variance = variance_floor(p00 + spec.measurement_variance);
        const double k0 = p00 / innovation_variance;
        const double k1 = p01 / innovation_variance;

        model.price = predicted_price + k0 * innovation;
        model.velocity = predicted_velocity + k1 * innovation;
        model.p00 = variance_floor((1.0 - k0) * p00);
        model.p01 = (1.0 - k0) * p01;
        model.p11 = variance_floor(p11 - k1 * p01);

        return -0.5 * (std::log(2.0 * pi_ * innovation_variance) + innovation * innovation / innovation_variance);
    }

    void update_probabilities(const std::array<double, model_count_> &predicted_probabilities,
                              const std::array<double, model_count_> &log_likelihoods,
                              double max_log_likelihood) {
        double total = 0.0;
        std::array<double, model_count_> next{};
        if (!std::isfinite(max_log_likelihood)) {
            probabilities_ = predicted_probabilities;
            normalize_probabilities();
            return;
        }
        for (std::size_t i = 0; i < model_count_; ++i) {
            next[i] = predicted_probabilities[i] * std::exp(log_likelihoods[i] - max_log_likelihood);
            total += next[i];
        }
        if (!std::isfinite(total) || total <= 0.0) {
            probabilities_ = predicted_probabilities;
            normalize_probabilities();
            return;
        }
        for (std::size_t i = 0; i < model_count_; ++i) {
            probabilities_[i] = next[i] / total;
        }
    }

    void normalize_probabilities() {
        double total = 0.0;
        for (double probability : probabilities_) {
            total += probability;
        }
        if (!std::isfinite(total) || total <= 0.0) {
            for (double &probability : probabilities_) {
                probability = 1.0 / static_cast<double>(model_count_);
            }
            return;
        }
        for (double &probability : probabilities_) {
            probability /= total;
        }
    }

    void update_combined_output() {
        double value = 0.0;
        double velocity = 0.0;
        for (std::size_t i = 0; i < model_count_; ++i) {
            value += probabilities_[i] * models_[i].price;
            velocity += probabilities_[i] * models_[i].velocity;
        }
        last_ = InteractingMultipleModelFilterResult{
            value,
            velocity,
            probabilities_[0],
            probabilities_[1],
            probabilities_[2],
            probabilities_[3],
        };
    }

    InteractingMultipleModelFilterResult output_for_warmup() const {
        if (!fillna_ && count_ < 2) {
            return InteractingMultipleModelFilterResult{nan(), nan(), nan(), nan(), nan(), nan()};
        }
        return last_;
    }

    double initial_price_;
    double initial_velocity_;
    double dt_;
    double stickiness_;
    std::array<ModelSpec, model_count_> specs_;
    std::array<ModelState, model_count_> models_;
    std::array<double, model_count_> probabilities_;
    bool initialized_;
    std::size_t count_;
    bool fillna_;
    InteractingMultipleModelFilterResult last_;
};

class ParticleFilterTrend {
public:
    ParticleFilterTrend(int particles = 128,
                        double initial_price = nan(),
                        double initial_velocity = 0.0,
                        double dt = 1.0,
                        double process_position_scale = 0.05,
                        double process_velocity_scale = 0.01,
                        double measurement_scale = 0.5,
                        double resample_threshold = 0.5,
                        std::uint64_t seed = 1,
                        bool fillna = true)
        : particles_(static_cast<std::size_t>(std::max(particles, 8))),
          weights_(particles_.size(), 1.0 / static_cast<double>(particles_.size())),
          initial_price_(initial_price),
          initial_velocity_(std::isfinite(initial_velocity) ? initial_velocity : 0.0),
          dt_(dt > 0.0 ? dt : 1.0),
          process_position_scale_(positive_scale(process_position_scale, 0.05)),
          process_velocity_scale_(positive_scale(process_velocity_scale, 0.01)),
          measurement_scale_(positive_scale(measurement_scale, 0.5)),
          resample_threshold_(std::clamp(std::isfinite(resample_threshold) ? resample_threshold : 0.5, 0.0, 1.0)),
          rng_(seed == 0 ? 1 : seed),
          initialized_(false),
          count_(0),
          fillna_(fillna),
          last_{nan(), nan(), nan(), nan()} {}

    ParticleFilterTrendResult update(double close) {
        if (!initialized_) {
            initialize(std::isfinite(initial_price_) ? initial_price_ : close);
            if (!std::isfinite(initial_price_)) {
                update_output(close, static_cast<double>(particles_.size()));
                ++count_;
                return output_for_warmup();
            }
        }

        propagate_and_weight(close);
        const double effective_sample_size = effective_sample_size_value();
        update_output(close, effective_sample_size);
        if (effective_sample_size < resample_threshold_ * static_cast<double>(particles_.size())) {
            resample();
        }
        ++count_;
        return output_for_warmup();
    }

    void advance(double close) {
        (void) update(close);
    }

    const ParticleFilterTrendResult &last() const {
        return last_;
    }

    template <typename Array>
    ParticleFilterTrendBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> trend(size);
        std::vector<double> velocity(size);
        std::vector<double> signal(size);
        std::vector<double> effective_sample_size(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const ParticleFilterTrendResult out = update(static_cast<double>(values[i]));
            trend[i] = out.trend;
            velocity[i] = out.velocity;
            signal[i] = out.signal;
            effective_sample_size[i] = out.effective_sample_size;
        }
        return {
            make_array(std::move(trend)),
            make_array(std::move(velocity)),
            make_array(std::move(signal)),
            make_array(std::move(effective_sample_size)),
        };
    }

private:
    struct Particle {
        double price;
        double velocity;
    };

    static constexpr double pi_ = 3.141592653589793238462643383279502884;

    static double positive_scale(double value, double fallback) {
        return (!std::isfinite(value) || value <= 0.0) ? fallback : value;
    }

    double uniform_open() {
        rng_ ^= rng_ >> 12;
        rng_ ^= rng_ << 25;
        rng_ ^= rng_ >> 27;
        const std::uint64_t value = rng_ * 2685821657736338717ULL;
        return (static_cast<double>(value >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }

    double standard_normal() {
        const double u1 = std::max(uniform_open(), 1.0e-16);
        const double u2 = uniform_open();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * pi_ * u2);
    }

    void initialize(double close) {
        const double spread = 0.25 * measurement_scale_;
        const double velocity_spread = process_velocity_scale_;
        const double equal_weight = 1.0 / static_cast<double>(particles_.size());
        for (std::size_t i = 0; i < particles_.size(); ++i) {
            particles_[i] = Particle{
                close + spread * standard_normal(),
                initial_velocity_ + velocity_spread * standard_normal(),
            };
            weights_[i] = equal_weight;
        }
        initialized_ = true;
    }

    void propagate_and_weight(double close) {
        std::vector<double> log_weights(particles_.size());
        double max_log_weight = -std::numeric_limits<double>::infinity();
        const double sqrt_dt = std::sqrt(dt_);
        for (std::size_t i = 0; i < particles_.size(); ++i) {
            Particle &particle = particles_[i];
            particle.velocity += process_velocity_scale_ * sqrt_dt * standard_normal();
            particle.price += particle.velocity * dt_ + process_position_scale_ * sqrt_dt * standard_normal();
            const double innovation = close - particle.price;
            const double log_weight = std::log(std::max(weights_[i], 1.0e-300)) - std::abs(innovation) / measurement_scale_;
            log_weights[i] = log_weight;
            max_log_weight = std::max(max_log_weight, log_weight);
        }

        double total = 0.0;
        for (std::size_t i = 0; i < particles_.size(); ++i) {
            weights_[i] = std::exp(log_weights[i] - max_log_weight);
            total += weights_[i];
        }
        if (!std::isfinite(total) || total <= 0.0) {
            const double equal_weight = 1.0 / static_cast<double>(particles_.size());
            for (double &weight : weights_) {
                weight = equal_weight;
            }
            return;
        }
        for (double &weight : weights_) {
            weight /= total;
        }
    }

    double effective_sample_size_value() const {
        double sum_squares = 0.0;
        for (double weight : weights_) {
            sum_squares += weight * weight;
        }
        return sum_squares > 0.0 ? 1.0 / sum_squares : static_cast<double>(particles_.size());
    }

    void update_output(double close, double effective_sample_size) {
        double trend = 0.0;
        double velocity = 0.0;
        for (std::size_t i = 0; i < particles_.size(); ++i) {
            trend += weights_[i] * particles_[i].price;
            velocity += weights_[i] * particles_[i].velocity;
        }
        const double signal = close > trend ? 1.0 : (close < trend ? -1.0 : 0.0);
        last_ = ParticleFilterTrendResult{trend, velocity, signal, effective_sample_size};
    }

    void resample() {
        std::vector<Particle> resampled;
        resampled.reserve(particles_.size());
        const double step = 1.0 / static_cast<double>(particles_.size());
        double u = uniform_open() * step;
        double cumulative = weights_[0];
        std::size_t index = 0;
        for (std::size_t i = 0; i < particles_.size(); ++i) {
            while (u > cumulative && index + 1 < particles_.size()) {
                ++index;
                cumulative += weights_[index];
            }
            resampled.push_back(particles_[index]);
            u += step;
        }
        particles_ = std::move(resampled);
        const double equal_weight = 1.0 / static_cast<double>(particles_.size());
        for (double &weight : weights_) {
            weight = equal_weight;
        }
    }

    ParticleFilterTrendResult output_for_warmup() const {
        if (!fillna_ && count_ < 2) {
            return ParticleFilterTrendResult{nan(), nan(), nan(), nan()};
        }
        return last_;
    }

    std::vector<Particle> particles_;
    std::vector<double> weights_;
    double initial_price_;
    double initial_velocity_;
    double dt_;
    double process_position_scale_;
    double process_velocity_scale_;
    double measurement_scale_;
    double resample_threshold_;
    std::uint64_t rng_;
    bool initialized_;
    std::size_t count_;
    bool fillna_;
    ParticleFilterTrendResult last_;
};

class SavitzkyGolayFilter {
public:
    SavitzkyGolayFilter(int window = 7, int polynomial_order = 2, double dt = 1.0, bool fillna = true)
        : window_(std::max(window, 3)),
          polynomial_order_(std::clamp(polynomial_order, 2, std::max(window_ - 1, 2))),
          dt_(dt > 0.0 ? dt : 1.0),
          values_(window_),
          smooth_coefficients_(coefficients(window_, polynomial_order_, 0, dt_)),
          first_derivative_coefficients_(coefficients(window_, polynomial_order_, 1, dt_)),
          second_derivative_coefficients_(coefficients(window_, polynomial_order_, 2, dt_)),
          fillna_(fillna),
          last_{nan(), nan(), nan()} {}

    SavitzkyGolayFilterResult update(double close) {
        values_.push(close);
        if (!fillna_ && !values_.full()) {
            last_ = SavitzkyGolayFilterResult{nan(), nan(), nan()};
            return last_;
        }

        const std::size_t size = values_.size();
        if (size == static_cast<std::size_t>(window_)) {
            last_ = SavitzkyGolayFilterResult{
                apply(smooth_coefficients_),
                apply(first_derivative_coefficients_),
                apply(second_derivative_coefficients_),
            };
            return last_;
        }

        if (size == 1) {
            last_ = SavitzkyGolayFilterResult{values_.at(0), 0.0, 0.0};
            return last_;
        }

        const int partial_order = std::min<int>(polynomial_order_, static_cast<int>(size) - 1);
        last_ = SavitzkyGolayFilterResult{
            apply(coefficients(static_cast<int>(size), partial_order, 0, dt_)),
            apply(coefficients(static_cast<int>(size), partial_order, 1, dt_)),
            partial_order >= 2 ? apply(coefficients(static_cast<int>(size), partial_order, 2, dt_)) : 0.0,
        };
        return last_;
    }

    void advance(double close) {
        (void) update(close);
    }

    const SavitzkyGolayFilterResult &last() const {
        return last_;
    }

    template <typename Array>
    SavitzkyGolayFilterBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> smooth(size);
        std::vector<double> first_derivative(size);
        std::vector<double> second_derivative(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const SavitzkyGolayFilterResult out = update(static_cast<double>(values[i]));
            smooth[i] = out.smooth;
            first_derivative[i] = out.first_derivative;
            second_derivative[i] = out.second_derivative;
        }
        return {
            make_array(std::move(smooth)),
            make_array(std::move(first_derivative)),
            make_array(std::move(second_derivative)),
        };
    }

private:
    static double factorial(int value) {
        double output = 1.0;
        for (int i = 2; i <= value; ++i) {
            output *= static_cast<double>(i);
        }
        return output;
    }

    static std::vector<double> invert(std::vector<double> matrix, int size) {
        std::vector<double> augmented(static_cast<std::size_t>(size * size * 2), 0.0);
        const int stride = 2 * size;
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                augmented[static_cast<std::size_t>(row * stride + col)] = matrix[static_cast<std::size_t>(row * size + col)];
            }
            augmented[static_cast<std::size_t>(row * stride + size + row)] = 1.0;
        }

        for (int col = 0; col < size; ++col) {
            int pivot = col;
            double pivot_abs = std::abs(augmented[static_cast<std::size_t>(pivot * stride + col)]);
            for (int row = col + 1; row < size; ++row) {
                const double candidate = std::abs(augmented[static_cast<std::size_t>(row * stride + col)]);
                if (candidate > pivot_abs) {
                    pivot = row;
                    pivot_abs = candidate;
                }
            }
            if (pivot_abs < 1.0e-14) {
                throw nb::value_error("SavitzkyGolayFilter polynomial fit is singular");
            }
            if (pivot != col) {
                for (int item = 0; item < stride; ++item) {
                    std::swap(
                        augmented[static_cast<std::size_t>(col * stride + item)],
                        augmented[static_cast<std::size_t>(pivot * stride + item)]);
                }
            }

            const double divisor = augmented[static_cast<std::size_t>(col * stride + col)];
            for (int item = 0; item < stride; ++item) {
                augmented[static_cast<std::size_t>(col * stride + item)] /= divisor;
            }
            for (int row = 0; row < size; ++row) {
                if (row == col) {
                    continue;
                }
                const double factor = augmented[static_cast<std::size_t>(row * stride + col)];
                if (factor == 0.0) {
                    continue;
                }
                for (int item = 0; item < stride; ++item) {
                    augmented[static_cast<std::size_t>(row * stride + item)] -= factor * augmented[static_cast<std::size_t>(col * stride + item)];
                }
            }
        }

        std::vector<double> inverse(static_cast<std::size_t>(size * size));
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                inverse[static_cast<std::size_t>(row * size + col)] = augmented[static_cast<std::size_t>(row * stride + size + col)];
            }
        }
        return inverse;
    }

    static std::vector<double> coefficients(int window, int polynomial_order, int derivative_order, double dt) {
        const int columns = std::min(polynomial_order, window - 1) + 1;
        std::vector<double> output(static_cast<std::size_t>(window), 0.0);
        if (derivative_order >= columns) {
            return output;
        }

        std::vector<double> normal(static_cast<std::size_t>(columns * columns), 0.0);
        std::vector<double> vandermonde(static_cast<std::size_t>(window * columns), 0.0);
        for (int row = 0; row < window; ++row) {
            const double x = static_cast<double>(row - (window - 1));
            double power = 1.0;
            for (int col = 0; col < columns; ++col) {
                vandermonde[static_cast<std::size_t>(row * columns + col)] = power;
                power *= x;
            }
        }

        for (int row = 0; row < columns; ++row) {
            for (int col = 0; col < columns; ++col) {
                double sum = 0.0;
                for (int sample = 0; sample < window; ++sample) {
                    sum += vandermonde[static_cast<std::size_t>(sample * columns + row)] *
                           vandermonde[static_cast<std::size_t>(sample * columns + col)];
                }
                normal[static_cast<std::size_t>(row * columns + col)] = sum;
            }
        }

        const std::vector<double> inverse = invert(std::move(normal), columns);
        const double scale = factorial(derivative_order) / std::pow(dt, derivative_order);
        for (int sample = 0; sample < window; ++sample) {
            double weight = 0.0;
            for (int col = 0; col < columns; ++col) {
                weight += inverse[static_cast<std::size_t>(derivative_order * columns + col)] *
                          vandermonde[static_cast<std::size_t>(sample * columns + col)];
            }
            output[static_cast<std::size_t>(sample)] = scale * weight;
        }
        return output;
    }

    double apply(const std::vector<double> &coefficients) const {
        double value = 0.0;
        for (std::size_t i = 0; i < coefficients.size(); ++i) {
            value += coefficients[i] * values_.at(i);
        }
        return value;
    }

    int window_;
    int polynomial_order_;
    double dt_;
    RollingBuffer values_;
    std::vector<double> smooth_coefficients_;
    std::vector<double> first_derivative_coefficients_;
    std::vector<double> second_derivative_coefficients_;
    bool fillna_;
    SavitzkyGolayFilterResult last_;
};

class NadarayaWatsonEnvelope {
public:
    NadarayaWatsonEnvelope(int window = 64, double bandwidth = 8.0, double multiplier = 2.0, bool fillna = true)
        : window_(std::max(window, 1)),
          bandwidth_(positive_scale(bandwidth, 8.0)),
          multiplier_(std::isfinite(multiplier) ? std::abs(multiplier) : 2.0),
          values_(window_),
          weights_(static_cast<std::size_t>(window_)),
          fillna_(fillna),
          last_{nan(), nan(), nan()} {
        for (int age = 0; age < window_; ++age) {
            const double scaled = static_cast<double>(age) / bandwidth_;
            weights_[static_cast<std::size_t>(age)] = std::exp(-0.5 * scaled * scaled);
        }
    }

    NadarayaWatsonEnvelopeResult update(double close) {
        values_.push(close);
        if (!fillna_ && !values_.full()) {
            last_ = NadarayaWatsonEnvelopeResult{nan(), nan(), nan()};
            return last_;
        }

        const std::size_t size = values_.size();
        double weighted_sum = 0.0;
        double weight_total = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            const std::size_t age = size - 1 - i;
            const double weight = weights_[age];
            weighted_sum += weight * values_.at(i);
            weight_total += weight;
        }
        const double middle = weight_total > 0.0 ? weighted_sum / weight_total : close;

        double variance = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            const std::size_t age = size - 1 - i;
            const double diff = values_.at(i) - middle;
            variance += weights_[age] * diff * diff;
        }
        variance = weight_total > 0.0 ? variance / weight_total : 0.0;
        const double band = multiplier_ * std::sqrt(std::max(variance, 0.0));
        last_ = NadarayaWatsonEnvelopeResult{middle, middle + band, middle - band};
        return last_;
    }

    void advance(double close) {
        (void) update(close);
    }

    const NadarayaWatsonEnvelopeResult &last() const {
        return last_;
    }

    template <typename Array>
    NadarayaWatsonEnvelopeBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> middle(size);
        std::vector<double> upper(size);
        std::vector<double> lower(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const NadarayaWatsonEnvelopeResult out = update(static_cast<double>(values[i]));
            middle[i] = out.middle;
            upper[i] = out.upper;
            lower[i] = out.lower;
        }
        return {
            make_array(std::move(middle)),
            make_array(std::move(upper)),
            make_array(std::move(lower)),
        };
    }

private:
    static double positive_scale(double value, double fallback) {
        return (!std::isfinite(value) || value <= 0.0) ? fallback : value;
    }

    int window_;
    double bandwidth_;
    double multiplier_;
    RollingBuffer values_;
    std::vector<double> weights_;
    bool fillna_;
    NadarayaWatsonEnvelopeResult last_;
};

class GaussianProcessRegressionBands {
public:
    GaussianProcessRegressionBands(int window = 32,
                                   double length_scale = 8.0,
                                   double signal_variance = 1.0,
                                   double noise_variance = 0.25,
                                   double multiplier = 2.0,
                                   bool fillna = true)
        : window_(std::max(window, 1)),
          length_scale_(positive_scale(length_scale, 8.0)),
          signal_variance_(positive_scale(signal_variance, 1.0)),
          noise_variance_(positive_scale(noise_variance, 0.25)),
          multiplier_(std::isfinite(multiplier) ? std::abs(multiplier) : 2.0),
          values_(window_),
          fillna_(fillna),
          last_{nan(), nan(), nan()} {}

    GaussianProcessRegressionBandsResult update(double close) {
        values_.push(close);
        if (!fillna_ && !values_.full()) {
            last_ = GaussianProcessRegressionBandsResult{nan(), nan(), nan()};
            return last_;
        }

        const std::size_t size = values_.size();
        double mean = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            mean += values_.at(i);
        }
        mean /= static_cast<double>(size);

        std::vector<double> matrix(size * size);
        std::vector<double> target(size);
        std::vector<double> centered(size);
        for (std::size_t row = 0; row < size; ++row) {
            const double x_row = static_cast<double>(row) - static_cast<double>(size - 1);
            target[row] = kernel(x_row, 0.0);
            centered[row] = values_.at(row) - mean;
            for (std::size_t col = 0; col < size; ++col) {
                const double x_col = static_cast<double>(col) - static_cast<double>(size - 1);
                matrix[row * size + col] = kernel(x_row, x_col);
            }
            matrix[row * size + row] += noise_variance_;
        }

        const std::vector<double> alpha = solve_linear(matrix, centered, size);
        const std::vector<double> variance_weights = solve_linear(std::move(matrix), target, size);

        double posterior_mean = mean;
        double explained_variance = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            posterior_mean += target[i] * alpha[i];
            explained_variance += target[i] * variance_weights[i];
        }

        const double posterior_variance = std::max(signal_variance_ - explained_variance, 0.0);
        const double band = multiplier_ * std::sqrt(posterior_variance);
        last_ = GaussianProcessRegressionBandsResult{posterior_mean, posterior_mean + band, posterior_mean - band};
        return last_;
    }

    void advance(double close) {
        (void) update(close);
    }

    const GaussianProcessRegressionBandsResult &last() const {
        return last_;
    }

    template <typename Array>
    GaussianProcessRegressionBandsBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> middle(size);
        std::vector<double> upper(size);
        std::vector<double> lower(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const GaussianProcessRegressionBandsResult out = update(static_cast<double>(values[i]));
            middle[i] = out.middle;
            upper[i] = out.upper;
            lower[i] = out.lower;
        }
        return {
            make_array(std::move(middle)),
            make_array(std::move(upper)),
            make_array(std::move(lower)),
        };
    }

private:
    static double positive_scale(double value, double fallback) {
        return (!std::isfinite(value) || value <= 0.0) ? fallback : value;
    }

    double kernel(double left, double right) const {
        const double scaled = (left - right) / length_scale_;
        return signal_variance_ * std::exp(-0.5 * scaled * scaled);
    }

    static std::vector<double> solve_linear(std::vector<double> matrix, std::vector<double> rhs, std::size_t size) {
        for (std::size_t col = 0; col < size; ++col) {
            std::size_t pivot = col;
            double pivot_abs = std::abs(matrix[pivot * size + col]);
            for (std::size_t row = col + 1; row < size; ++row) {
                const double candidate = std::abs(matrix[row * size + col]);
                if (candidate > pivot_abs) {
                    pivot = row;
                    pivot_abs = candidate;
                }
            }
            if (pivot_abs < 1.0e-14) {
                throw nb::value_error("GaussianProcessRegressionBands kernel matrix is singular");
            }
            if (pivot != col) {
                for (std::size_t item = col; item < size; ++item) {
                    std::swap(matrix[col * size + item], matrix[pivot * size + item]);
                }
                std::swap(rhs[col], rhs[pivot]);
            }

            const double divisor = matrix[col * size + col];
            for (std::size_t item = col; item < size; ++item) {
                matrix[col * size + item] /= divisor;
            }
            rhs[col] /= divisor;

            for (std::size_t row = 0; row < size; ++row) {
                if (row == col) {
                    continue;
                }
                const double factor = matrix[row * size + col];
                if (factor == 0.0) {
                    continue;
                }
                for (std::size_t item = col; item < size; ++item) {
                    matrix[row * size + item] -= factor * matrix[col * size + item];
                }
                rhs[row] -= factor * rhs[col];
            }
        }
        return rhs;
    }

    int window_;
    double length_scale_;
    double signal_variance_;
    double noise_variance_;
    double multiplier_;
    RollingBuffer values_;
    bool fillna_;
    GaussianProcessRegressionBandsResult last_;
};

class ZigZagSwingDetector {
public:
    ZigZagSwingDetector(double percent_change = 5.0)
        : threshold_(threshold_fraction(percent_change)),
          initialized_(false),
          index_(0),
          direction_(0.0),
          pivot_(0.0),
          pivot_index_(0),
          start_(0.0),
          extreme_(0.0),
          extreme_index_(0),
          last_{nan(), 0.0, nan(), nan()} {}

    ZigZagSwingDetectorResult update(double close) {
        if (!initialized_) {
            initialized_ = true;
            pivot_ = close;
            start_ = close;
            extreme_ = close;
            last_ = ZigZagSwingDetectorResult{close, 0.0, close, 0.0};
            ++index_;
            return last_;
        }

        double confirmed_pivot = pivot_;
        double confirmed_pivot_index = static_cast<double>(pivot_index_);
        if (direction_ == 0.0) {
            if (close >= start_ * (1.0 + threshold_)) {
                direction_ = 1.0;
                pivot_ = start_;
                pivot_index_ = 0;
                extreme_ = close;
                extreme_index_ = index_;
            } else if (close <= start_ * (1.0 - threshold_)) {
                direction_ = -1.0;
                pivot_ = start_;
                pivot_index_ = 0;
                extreme_ = close;
                extreme_index_ = index_;
            } else {
                if (std::abs(close - start_) > std::abs(extreme_ - start_)) {
                    extreme_ = close;
                    extreme_index_ = index_;
                }
            }
            confirmed_pivot = pivot_;
            confirmed_pivot_index = static_cast<double>(pivot_index_);
        } else if (direction_ > 0.0) {
            if (close >= extreme_) {
                extreme_ = close;
                extreme_index_ = index_;
            } else if (close <= extreme_ * (1.0 - threshold_)) {
                pivot_ = extreme_;
                pivot_index_ = extreme_index_;
                direction_ = -1.0;
                extreme_ = close;
                extreme_index_ = index_;
                confirmed_pivot = pivot_;
                confirmed_pivot_index = static_cast<double>(pivot_index_);
            }
        } else {
            if (close <= extreme_) {
                extreme_ = close;
                extreme_index_ = index_;
            } else if (close >= extreme_ * (1.0 + threshold_)) {
                pivot_ = extreme_;
                pivot_index_ = extreme_index_;
                direction_ = 1.0;
                extreme_ = close;
                extreme_index_ = index_;
                confirmed_pivot = pivot_;
                confirmed_pivot_index = static_cast<double>(pivot_index_);
            }
        }

        last_ = ZigZagSwingDetectorResult{extreme_, direction_, confirmed_pivot, confirmed_pivot_index};
        ++index_;
        return last_;
    }

    void advance(double close) {
        (void) update(close);
    }

    const ZigZagSwingDetectorResult &last() const {
        return last_;
    }

    template <typename Array>
    ZigZagSwingDetectorBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> value(size);
        std::vector<double> direction(size);
        std::vector<double> pivot(size);
        std::vector<double> pivot_index(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const ZigZagSwingDetectorResult out = update(static_cast<double>(values[i]));
            value[i] = out.value;
            direction[i] = out.direction;
            pivot[i] = out.pivot;
            pivot_index[i] = out.pivot_index;
        }
        return {
            make_array(std::move(value)),
            make_array(std::move(direction)),
            make_array(std::move(pivot)),
            make_array(std::move(pivot_index)),
        };
    }

private:
    static double threshold_fraction(double percent_change) {
        if (!std::isfinite(percent_change) || percent_change <= 0.0) {
            return 0.05;
        }
        return percent_change > 1.0 ? percent_change / 100.0 : percent_change;
    }

    double threshold_;
    bool initialized_;
    std::size_t index_;
    double direction_;
    double pivot_;
    std::size_t pivot_index_;
    double start_;
    double extreme_;
    std::size_t extreme_index_;
    ZigZagSwingDetectorResult last_;
};

class RenkoBrickGenerator {
public:
    RenkoBrickGenerator(double brick_size = 1.0)
        : brick_size_((std::isfinite(brick_size) && brick_size > 0.0) ? brick_size : 1.0),
          initialized_(false),
          brick_close_(0.0),
          direction_(0.0),
          last_{nan(), nan(), 0.0, 0.0, 0.0} {}

    RenkoBrickGeneratorResult update(double close) {
        if (!initialized_) {
            initialized_ = true;
            brick_close_ = close;
            last_ = RenkoBrickGeneratorResult{close, close, 0.0, 0.0, 0.0};
            return last_;
        }

        const double brick_open = brick_close_;
        const double previous_direction = direction_;
        double signed_bricks = 0.0;
        while (close >= brick_close_ + brick_size_) {
            brick_close_ += brick_size_;
            direction_ = 1.0;
            signed_bricks += 1.0;
        }
        while (close <= brick_close_ - brick_size_) {
            brick_close_ -= brick_size_;
            direction_ = -1.0;
            signed_bricks -= 1.0;
        }
        const double reversal = signed_bricks != 0.0 && previous_direction != 0.0 && direction_ != previous_direction ? 1.0 : 0.0;
        last_ = RenkoBrickGeneratorResult{brick_open, brick_close_, direction_, signed_bricks, reversal};
        return last_;
    }

    void advance(double close) {
        (void) update(close);
    }

    const RenkoBrickGeneratorResult &last() const {
        return last_;
    }

    template <typename Array>
    RenkoBrickGeneratorBatchResult batch_array(const Array &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> brick_open(size);
        std::vector<double> brick_close(size);
        std::vector<double> direction(size);
        std::vector<double> bricks(size);
        std::vector<double> reversal(size);
        const auto *values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const RenkoBrickGeneratorResult out = update(static_cast<double>(values[i]));
            brick_open[i] = out.brick_open;
            brick_close[i] = out.brick_close;
            direction[i] = out.direction;
            bricks[i] = out.bricks;
            reversal[i] = out.reversal;
        }
        return {
            make_array(std::move(brick_open)),
            make_array(std::move(brick_close)),
            make_array(std::move(direction)),
            make_array(std::move(bricks)),
            make_array(std::move(reversal)),
        };
    }

private:
    double brick_size_;
    bool initialized_;
    double brick_close_;
    double direction_;
    RenkoBrickGeneratorResult last_;
};

class HeikinAshiTransform {
public:
    HeikinAshiTransform()
        : initialized_(false),
          previous_open_(0.0),
          previous_close_(0.0),
          last_{nan(), nan(), nan(), nan()} {}

    HeikinAshiTransformResult update(double open, double high, double low, double close) {
        const double ha_close = 0.25 * (open + high + low + close);
        const double ha_open = initialized_ ? 0.5 * (previous_open_ + previous_close_) : 0.5 * (open + close);
        const double ha_high = std::max(high, std::max(ha_open, ha_close));
        const double ha_low = std::min(low, std::min(ha_open, ha_close));
        previous_open_ = ha_open;
        previous_close_ = ha_close;
        initialized_ = true;
        last_ = HeikinAshiTransformResult{ha_open, ha_high, ha_low, ha_close};
        return last_;
    }

    void advance(double open, double high, double low, double close) {
        (void) update(open, high, low, close);
    }

    const HeikinAshiTransformResult &last() const {
        return last_;
    }

    template <typename Array0, typename Array1, typename Array2, typename Array3>
    HeikinAshiTransformBatchResult batch_array(const Array0 &open, const Array1 &high, const Array2 &low, const Array3 &close) {
        const std::size_t size = open.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        require_same_size(size, close.shape(0));
        std::vector<double> out_open(size);
        std::vector<double> out_high(size);
        std::vector<double> out_low(size);
        std::vector<double> out_close(size);
        const auto *open_values = open.data();
        const auto *high_values = high.data();
        const auto *low_values = low.data();
        const auto *close_values = close.data();
        for (std::size_t i = 0; i < size; ++i) {
            const HeikinAshiTransformResult out = update(
                static_cast<double>(open_values[i]),
                static_cast<double>(high_values[i]),
                static_cast<double>(low_values[i]),
                static_cast<double>(close_values[i]));
            out_open[i] = out.open;
            out_high[i] = out.high;
            out_low[i] = out.low;
            out_close[i] = out.close;
        }
        return {
            make_array(std::move(out_open)),
            make_array(std::move(out_high)),
            make_array(std::move(out_low)),
            make_array(std::move(out_close)),
        };
    }

private:
    bool initialized_;
    double previous_open_;
    double previous_close_;
    HeikinAshiTransformResult last_;
};

class High {
public:
    explicit High(int window = 1, bool fillna = true)
        : values_(window, true),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double value) {
        values_.push(value);
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : values_.value();
    }

    template <typename Array>
    nb::object batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        if (counter_ == 0 && static_cast<std::size_t>(std::max(window_, 1)) <= SMALL_SCAN_WINDOW_LIMIT) {
            batch_kernels::rolling_high_small_window_scan(
                input.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                fillna_,
                true,
                output.data());
            batch_kernels::rebuild_extreme_state(input.data(), size, static_cast<std::size_t>(std::max(window_, 1)), values_);
            counter_ = static_cast<long>(size);
        } else {
            batch_kernels::rolling_extreme_value(input.data(), size, values_, window_, fillna_, counter_, output.data());
        }
        return make_array(std::move(output));
    }

private:
    RollingExtreme values_;
    int window_;
    bool fillna_;
    long counter_;
};

class Low {
public:
    explicit Low(int window = 1, bool fillna = true)
        : values_(window, false),
          window_(window),
          fillna_(fillna),
          counter_(0) {}

    double update(double value) {
        values_.push(value);
        const bool return_nan = !fillna_ && counter_ < window_;
        ++counter_;
        return return_nan ? nan() : values_.value();
    }

    template <typename Array>
    nb::object batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        if (counter_ == 0 && static_cast<std::size_t>(std::max(window_, 1)) <= SMALL_SCAN_WINDOW_LIMIT) {
            batch_kernels::rolling_low_small_window_scan(
                input.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                fillna_,
                true,
                output.data());
            batch_kernels::rebuild_extreme_state(input.data(), size, static_cast<std::size_t>(std::max(window_, 1)), values_);
            counter_ = static_cast<long>(size);
        } else {
            batch_kernels::rolling_extreme_value(input.data(), size, values_, window_, fillna_, counter_, output.data());
        }
        return make_array(std::move(output));
    }

private:
    RollingExtreme values_;
    int window_;
    bool fillna_;
    long counter_;
};

class StochRSI {
public:
    StochRSI(int window = 14, bool fillna = true)
        : rsi_(window, true),
          lowest_rsi_(window, fillna),
          highest_rsi_(window, fillna),
          window_size_(window),
          fillna_(fillna) {}

    double update(double value) {
        const double rsi = rsi_.update(value);
        const double low = lowest_rsi_.update(rsi);
        const double high = highest_rsi_.update(rsi);
        if (high == low) {
            return 0.0;
        }
        return (rsi / low) / (high - low);
    }

private:
    RSI rsi_;
    Low lowest_rsi_;
    High highest_rsi_;
    int window_size_;
    bool fillna_;
};

class FastStochastic {
public:
    FastStochastic(int fastk = 5, int fastd = 3, bool fillna = true)
        : highs_(fastk, true),
          lows_(fastk, false),
          fastd_(fastd, fillna),
          fastk_window_(fastk),
          fillna_(fillna),
          last_{nan(), nan()} {}

    FastStochasticResult update(double close, double high, double low) {
        update_core(close, high, low);
        return last_;
    }

    void advance(double close, double high, double low) {
        update_core(close, high, low);
    }

    inline const FastStochasticResult &last() const {
        return last_;
    }

private:
    inline void update_core(double close, double high, double low) {
        highs_.push(high);
        lows_.push(low);
        const double fastk = (!fillna_ && !highs_.full()) ? nan() :
            safe_divide(100.0 * (close - lows_.value()), highs_.value() - lows_.value());
        const double fastd = fastd_.update(fastk);
        last_ = {fastk, fastd};
    }

public:
    template <typename Array0, typename Array1, typename Array2>
    FastStochasticBatchResult batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> fastk(size);
        std::vector<double> fastd(size);
        if (highs_.size() == 0 && static_cast<std::size_t>(std::max(fastk_window_, 1)) <= SMALL_SCAN_WINDOW_LIMIT) {
            batch_kernels::stochastic_fastk_small_window_scan(
                close.data(),
                high.data(),
                low.data(),
                size,
                static_cast<std::size_t>(std::max(fastk_window_, 1)),
                fillna_,
                fastk.data());
            for (std::size_t i = 0; i < size; ++i) {
                fastd[i] = fastd_.update(fastk[i]);
            }
            batch_kernels::rebuild_pair_extreme_state(
                high.data(),
                low.data(),
                size,
                static_cast<std::size_t>(std::max(fastk_window_, 1)),
                highs_,
                lows_);
        } else {
            batch_kernels::fast_stochastic(
                close.data(),
                high.data(),
                low.data(),
                size,
                highs_,
                lows_,
                fastd_,
                fillna_,
                fastk.data(),
                fastd.data());
        }
        return {make_array(std::move(fastk)), make_array(std::move(fastd))};
    }

private:
    RollingExtreme highs_;
    RollingExtreme lows_;
    SMA fastd_;
    int fastk_window_;
    bool fillna_;
    FastStochasticResult last_;
};

class Stochastic {
public:
    Stochastic(int fastk = 5, int slowk = 3, int slowd = 3, bool fillna = true)
        : highs_(fastk, true),
          lows_(fastk, false),
          slowk_(slowk, fillna),
          slowd_(slowd, fillna),
          fastk_window_(fastk),
          fillna_(fillna),
          last_{nan(), nan()} {}

    StochasticResult update(double close, double high, double low) {
        update_core(close, high, low);
        return last_;
    }

    void advance(double close, double high, double low) {
        update_core(close, high, low);
    }

    inline const StochasticResult &last() const {
        return last_;
    }

private:
    inline void update_core(double close, double high, double low) {
        highs_.push(high);
        lows_.push(low);
        const double fastk = (!fillna_ && !highs_.full()) ? nan() :
            safe_divide(100.0 * (close - lows_.value()), highs_.value() - lows_.value());
        const double slowk = slowk_.update(fastk);
        const double slowd = slowd_.update(slowk);
        last_ = {slowk, slowd};
    }

public:
    template <typename Array0, typename Array1, typename Array2>
    StochasticBatchResult batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> slowk(size);
        std::vector<double> slowd(size);
        if (highs_.size() == 0 && static_cast<std::size_t>(std::max(fastk_window_, 1)) <= SMALL_SCAN_WINDOW_LIMIT) {
            std::vector<double> fastk(size);
            batch_kernels::stochastic_fastk_small_window_scan(
                close.data(),
                high.data(),
                low.data(),
                size,
                static_cast<std::size_t>(std::max(fastk_window_, 1)),
                fillna_,
                fastk.data());
            for (std::size_t i = 0; i < size; ++i) {
                slowk[i] = slowk_.update(fastk[i]);
                slowd[i] = slowd_.update(slowk[i]);
            }
            batch_kernels::rebuild_pair_extreme_state(
                high.data(),
                low.data(),
                size,
                static_cast<std::size_t>(std::max(fastk_window_, 1)),
                highs_,
                lows_);
        } else {
            batch_kernels::stochastic(
                close.data(),
                high.data(),
                low.data(),
                size,
                highs_,
                lows_,
                slowk_,
                slowd_,
                fillna_,
                slowk.data(),
                slowd.data());
        }
        return {make_array(std::move(slowk)), make_array(std::move(slowd))};
    }

private:
    RollingExtreme highs_;
    RollingExtreme lows_;
    SMA slowk_;
    SMA slowd_;
    int fastk_window_;
    bool fillna_;
    StochasticResult last_;
};

class WilliamsR {
public:
    WilliamsR(int window = 14, bool fillna = true)
        : highs_(window, true),
          lows_(window, false),
          window_(window),
          fillna_(fillna) {}

    double update(double close, double high, double low) {
        highs_.push(high);
        lows_.push(low);
        if (!fillna_ && !highs_.full()) {
            return nan();
        }
        const double highest = highs_.value();
        const double lowest = lows_.value();
        return safe_divide(-100.0 * (highest - close), highest - lowest);
    }

    template <typename Array0, typename Array1, typename Array2>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        if (highs_.size() == 0 && static_cast<std::size_t>(std::max(window_, 1)) <= SMALL_SCAN_WINDOW_LIMIT) {
            batch_kernels::williams_r_small_window_scan(
                close.data(),
                high.data(),
                low.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                fillna_,
                output.data());
            batch_kernels::rebuild_pair_extreme_state(
                high.data(),
                low.data(),
                size,
                static_cast<std::size_t>(std::max(window_, 1)),
                highs_,
                lows_);
        } else {
            batch_kernels::williams_r(close.data(), high.data(), low.data(), size, highs_, lows_, fillna_, output.data());
        }
        return make_array(std::move(output));
    }

private:
    RollingExtreme highs_;
    RollingExtreme lows_;
    int window_;
    bool fillna_;
};

class UltimateOscillator {
public:
    UltimateOscillator(int short_window = 7, int medium_window = 14, int long_window = 28, bool fillna = true)
        : bp_short_(short_window),
          tr_short_(short_window),
          bp_medium_(medium_window),
          tr_medium_(medium_window),
          bp_long_(long_window),
          tr_long_(long_window),
          bp_short_sum_(0.0),
          tr_short_sum_(0.0),
          bp_medium_sum_(0.0),
          tr_medium_sum_(0.0),
          bp_long_sum_(0.0),
          tr_long_sum_(0.0),
          previous_close_(0.0),
          first_(true),
          fillna_(fillna) {}

    double update(double close, double high, double low) {
        const double prev = first_ ? close : previous_close_;
        const double buying_pressure = close - std::min(low, prev);
        const double range = std::max(high, prev) - std::min(low, prev);
        previous_close_ = close;
        first_ = false;

        rolling_sum_push(bp_short_, bp_short_sum_, buying_pressure);
        rolling_sum_push(tr_short_, tr_short_sum_, range);
        rolling_sum_push(bp_medium_, bp_medium_sum_, buying_pressure);
        rolling_sum_push(tr_medium_, tr_medium_sum_, range);
        rolling_sum_push(bp_long_, bp_long_sum_, buying_pressure);
        rolling_sum_push(tr_long_, tr_long_sum_, range);

        if (!fillna_ && !bp_long_.full()) {
            return nan();
        }

        const double avg_short = safe_divide(bp_short_sum_, tr_short_sum_);
        const double avg_medium = safe_divide(bp_medium_sum_, tr_medium_sum_);
        const double avg_long = safe_divide(bp_long_sum_, tr_long_sum_);
        return 100.0 * (4.0 * avg_short + 2.0 * avg_medium + avg_long) / 7.0;
    }

    template <typename Array0, typename Array1, typename Array2>
    nb::object batch_array(const Array0 &close, const Array1 &high, const Array2 &low) {
        const std::size_t size = close.shape(0);
        require_same_size(size, high.shape(0));
        require_same_size(size, low.shape(0));
        std::vector<double> output(size);
        batch_kernels::ultimate_oscillator(
            close.data(),
            high.data(),
            low.data(),
            size,
            bp_short_,
            tr_short_,
            bp_medium_,
            tr_medium_,
            bp_long_,
            tr_long_,
            bp_short_sum_,
            tr_short_sum_,
            bp_medium_sum_,
            tr_medium_sum_,
            bp_long_sum_,
            tr_long_sum_,
            previous_close_,
            first_,
            fillna_,
            output.data());
        return make_array(std::move(output));
    }

private:
    RollingBuffer bp_short_;
    RollingBuffer tr_short_;
    RollingBuffer bp_medium_;
    RollingBuffer tr_medium_;
    RollingBuffer bp_long_;
    RollingBuffer tr_long_;
    double bp_short_sum_;
    double tr_short_sum_;
    double bp_medium_sum_;
    double tr_medium_sum_;
    double bp_long_sum_;
    double tr_long_sum_;
    double previous_close_;
    bool first_;
    bool fillna_;
};

class ParabolicSAR {
public:
    ParabolicSAR(double acceleration = 0.02, double maximum = 0.2)
        : acceleration_(acceleration),
          maximum_(maximum),
          acceleration_factor_(acceleration),
          extreme_point_(0.0),
          sar_(0.0),
          previous_high_(0.0),
          previous_low_(0.0),
          rising_(true),
          count_(0) {}

    double update(double high, double low) {
        if (count_ == 0) {
            previous_high_ = high;
            previous_low_ = low;
            sar_ = low;
            extreme_point_ = high;
            ++count_;
            return sar_;
        }

        if (count_ == 1) {
            rising_ = high >= previous_high_;
            sar_ = rising_ ? std::min(previous_low_, low) : std::max(previous_high_, high);
            extreme_point_ = rising_ ? std::max(previous_high_, high) : std::min(previous_low_, low);
            previous_high_ = high;
            previous_low_ = low;
            ++count_;
            return sar_;
        }

        double next_sar = sar_ + acceleration_factor_ * (extreme_point_ - sar_);

        if (rising_) {
            next_sar = std::min(next_sar, std::min(previous_low_, low));
            if (low < next_sar) {
                rising_ = false;
                next_sar = extreme_point_;
                extreme_point_ = low;
                acceleration_factor_ = acceleration_;
            } else if (high > extreme_point_) {
                extreme_point_ = high;
                acceleration_factor_ = std::min(acceleration_factor_ + acceleration_, maximum_);
            }
        } else {
            next_sar = std::max(next_sar, std::max(previous_high_, high));
            if (high > next_sar) {
                rising_ = true;
                next_sar = extreme_point_;
                extreme_point_ = high;
                acceleration_factor_ = acceleration_;
            } else if (low < extreme_point_) {
                extreme_point_ = low;
                acceleration_factor_ = std::min(acceleration_factor_ + acceleration_, maximum_);
            }
        }

        previous_high_ = high;
        previous_low_ = low;
        sar_ = next_sar;
        ++count_;
        return sar_;
    }

private:
    double acceleration_;
    double maximum_;
    double acceleration_factor_;
    double extreme_point_;
    double sar_;
    double previous_high_;
    double previous_low_;
    bool rising_;
    long count_;
};

class TSI {
public:
    TSI(int window_1 = 25, int window_2 = 13)
        : p_(1, true),
          a_(window_1, true),
          b_(window_2, true),
          d_(1, true),
          e_(window_1, true),
          f_(window_2, true) {}

    double update(double x) {
        return (100.0 * b_.update(a_.update(x - p_.update(x)))) /
               f_.update(e_.update(std::abs(x - d_.update(x))));
    }

private:
    Delay p_;
    EMA a_;
    EMA b_;
    Delay d_;
    EMA e_;
    EMA f_;
};

class DailyReturn {
public:
    explicit DailyReturn(bool fillna = true)
        : previous_close_(0.0),
          first_(true),
          fillna_(fillna) {}

    double update(double close) {
        if (first_) {
            previous_close_ = close;
            first_ = false;
            return fillna_ ? 0.0 : nan();
        }
        const double retval = safe_divide(close, previous_close_) * 100.0 - 100.0;
        previous_close_ = close;
        return retval;
    }

    nb::object batch(const InputArray &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(close(i));
        }
        return make_array(std::move(output));
    }

private:
    double previous_close_;
    bool first_;
    bool fillna_;
};

class DailyLogReturn {
public:
    explicit DailyLogReturn(bool fillna = true)
        : previous_close_(0.0),
          first_(true),
          fillna_(fillna) {}

    double update(double close) {
        if (first_) {
            previous_close_ = close;
            first_ = false;
            return fillna_ ? 0.0 : nan();
        }
        const double retval = (std::log(close) - std::log(previous_close_)) * 100.0;
        previous_close_ = close;
        return retval;
    }

    nb::object batch(const InputArray &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(close(i));
        }
        return make_array(std::move(output));
    }

private:
    double previous_close_;
    bool first_;
    bool fillna_;
};

class CumulativeReturn {
public:
    CumulativeReturn()
        : first_close_(0.0),
          first_(true) {}

    double update(double close) {
        if (first_) {
            first_close_ = close;
            first_ = false;
        }
        return safe_divide(close, first_close_) * 100.0 - 100.0;
    }

    nb::object batch(const InputArray &close) {
        const std::size_t size = close.shape(0);
        std::vector<double> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = update(close(i));
        }
        return make_array(std::move(output));
    }

private:
    double first_close_;
    bool first_;
};

class StdDev {
public:
    explicit StdDev(int window, bool fillna = true)
        : values_(window),
          fillna_(fillna),
          window_size_(window),
          counter_(0),
          sum_(0.0),
          sum2_(0.0) {}

    double update(double value) {
        const bool return_nan = !fillna_ && counter_ < window_size_;
        ++counter_;

        if (values_.full()) {
            const double oldest = values_.oldest();
            sum_ -= oldest;
            sum2_ -= oldest * oldest;
        }

        values_.push(value);
        sum_ += value;
        sum2_ += value * value;

        if (return_nan) {
            return nan();
        }

        const double n = static_cast<double>(values_.size());
        const double variance = (sum2_ - sum_ * sum_ / n) / n;
        return std::sqrt(variance < 0.0 && variance > -1e-12 ? 0.0 : variance);
    }

    template <typename Array>
    nb::object batch_array(const Array &input) {
        const std::size_t size = input.shape(0);
        std::vector<double> output(size);
        if (counter_ == 0 && values_.size() == 0) {
            const std::size_t window = values_.capacity();
            batch_kernels::rolling_stddev_fresh(input.data(), size, window, fillna_, window_size_, output.data());
            batch_kernels::rebuild_buffer_sum(input.data(), size, window, values_, sum_, sum2_);
            counter_ = static_cast<long>(size);
        } else {
            batch_kernels::rolling_stddev(input.data(), size, values_, fillna_, window_size_, counter_, sum_, sum2_, output.data());
        }
        return make_array(std::move(output));
    }

private:
    RollingBuffer values_;
    bool fillna_;
    int window_size_;
    long counter_;
    double sum_;
    double sum2_;
};

class BollingerBands {
public:
    BollingerBands(int window = 20, bool fillna = true)
        : sma_(window, fillna),
          stddev_(window, fillna),
          last_{nan(), nan(), nan()} {}

    BollingerBandsResult update(double value) {
        update_core(value);
        return last_;
    }

    void advance(double value) {
        update_core(value);
    }

    inline const BollingerBandsResult &last() const {
        return last_;
    }

private:
    inline void update_core(double value) {
        const double stddev = stddev_.update(value);
        const double middle = sma_.update(value);
        last_ = {middle, middle + stddev * 2.0, middle - stddev * 2.0};
    }

    SMA sma_;
    StdDev stddev_;
    BollingerBandsResult last_;
};

LinearRegressionBatchResult batch_linear_regression(LinearRegressionCore &indicator, const InputArray &input) {
    const std::size_t size = input.shape(0);
    std::vector<double> value(size);
    std::vector<double> slope(size);
    std::vector<double> intercept(size);
    std::vector<double> angle(size);
    std::vector<double> tsf(size);
    const double *values = input.data();

    for (std::size_t i = 0; i < size; ++i) {
        const LinearRegressionResult out = indicator.update(values[i]);
        value[i] = out.value;
        slope[i] = out.slope;
        intercept[i] = out.intercept;
        angle[i] = out.angle;
        tsf[i] = out.tsf;
    }

    return {
        make_array(std::move(value)),
        make_array(std::move(slope)),
        make_array(std::move(intercept)),
        make_array(std::move(angle)),
        make_array(std::move(tsf)),
    };
}

template <typename Array>
HighLowBatchResult batch_high_low(HighLow &indicator, const Array &input) {
    return indicator.batch_array(input);
}

template <typename Array>
HighLowIndexBatchResult batch_high_low_index(HighLowIndex &indicator, const Array &input) {
    return indicator.batch_array(input);
}

template <typename Array0, typename Array1>
AroonBatchResult batch_aroon(Aroon &indicator, const Array0 &high, const Array1 &low) {
    return indicator.batch_array(high, low);
}

template <typename Array0, typename Array1, typename Array2>
FastStochasticBatchResult batch_fast_stochastic(
    FastStochastic &indicator,
    const Array0 &close,
    const Array1 &high,
    const Array2 &low) {
    return indicator.batch_array(close, high, low);
}

template <typename Array0, typename Array1, typename Array2>
StochasticBatchResult batch_stochastic(
    Stochastic &indicator,
    const Array0 &close,
    const Array1 &high,
    const Array2 &low) {
    return indicator.batch_array(close, high, low);
}

template <typename Array>
BollingerBandsBatchResult batch_bollinger_bands(BollingerBands &indicator, const Array &input) {
    const std::size_t size = input.shape(0);
    std::vector<double> middle(size);
    std::vector<double> upper(size);
    std::vector<double> lower(size);
    const auto *values = input.data();

    for (std::size_t i = 0; i < size; ++i) {
        const BollingerBandsResult out = indicator.update(static_cast<double>(values[i]));
        middle[i] = out.middle;
        upper[i] = out.upper;
        lower[i] = out.lower;
    }

    return {make_array(std::move(middle)), make_array(std::move(upper)), make_array(std::move(lower))};
}

template <typename Indicator, typename Array0, typename Array1, typename Array2>
KeltnerChannelBatchResult batch_keltner(
    Indicator &indicator,
    const Array0 &close,
    const Array1 &high,
    const Array2 &low) {
    const std::size_t size = close.shape(0);
    require_same_size(size, high.shape(0));
    require_same_size(size, low.shape(0));
    std::vector<double> middle(size);
    std::vector<double> upper(size);
    std::vector<double> lower(size);
    const auto *close_values = close.data();
    const auto *high_values = high.data();
    const auto *low_values = low.data();

    for (std::size_t i = 0; i < size; ++i) {
        const KeltnerChannelResult out = indicator.update(
            static_cast<double>(close_values[i]),
            static_cast<double>(high_values[i]),
            static_cast<double>(low_values[i]));
        middle[i] = out.middle;
        upper[i] = out.upper;
        lower[i] = out.lower;
    }

    return {make_array(std::move(middle)), make_array(std::move(upper)), make_array(std::move(lower))};
}

template <typename Array0, typename Array1, typename Array2>
SuperTrendBatchResult batch_super_trend(SuperTrend &indicator, const Array0 &close, const Array1 &high, const Array2 &low) {
    return indicator.batch_array(close, high, low);
}

template <typename Array>
PercentagePriceBatchResult batch_percentage_price(PercentagePrice &indicator, const Array &close) {
    const std::size_t size = close.shape(0);
    std::vector<double> ppo(size);
    std::vector<double> signal(size);
    std::vector<double> histogram(size);
    const auto *values = close.data();

    for (std::size_t i = 0; i < size; ++i) {
        const PercentagePriceResult out = indicator.update(static_cast<double>(values[i]));
        ppo[i] = out.ppo;
        signal[i] = out.signal;
        histogram[i] = out.histogram;
    }

    return {make_array(std::move(ppo)), make_array(std::move(signal)), make_array(std::move(histogram))};
}

template <typename Array>
PercentageVolumeBatchResult batch_percentage_volume(PercentageVolume &indicator, const Array &volume) {
    const std::size_t size = volume.shape(0);
    std::vector<double> pvo(size);
    std::vector<double> signal(size);
    std::vector<double> histogram(size);
    const auto *values = volume.data();

    for (std::size_t i = 0; i < size; ++i) {
        const PercentageVolumeResult out = indicator.update(static_cast<double>(values[i]));
        pvo[i] = out.pvo;
        signal[i] = out.signal;
        histogram[i] = out.histogram;
    }

    return {make_array(std::move(pvo)), make_array(std::move(signal)), make_array(std::move(histogram))};
}

template <typename Array0, typename Array1, typename Array2>
DonchianChannelBatchResult batch_donchian(
    DonchianChannel &indicator,
    const Array0 &close,
    const Array1 &high,
    const Array2 &low) {
    const std::size_t size = close.shape(0);
    require_same_size(size, high.shape(0));
    require_same_size(size, low.shape(0));
    std::vector<double> upper(size);
    std::vector<double> lower(size);
    std::vector<double> middle(size);
    std::vector<double> width(size);
    std::vector<double> percent(size);
    const auto *close_values = close.data();
    const auto *high_values = high.data();
    const auto *low_values = low.data();

    for (std::size_t i = 0; i < size; ++i) {
        const DonchianChannelResult out = indicator.update(
            static_cast<double>(close_values[i]),
            static_cast<double>(high_values[i]),
            static_cast<double>(low_values[i]));
        upper[i] = out.upper;
        lower[i] = out.lower;
        middle[i] = out.middle;
        width[i] = out.width;
        percent[i] = out.percent;
    }

    return {
        make_array(std::move(upper)),
        make_array(std::move(lower)),
        make_array(std::move(middle)),
        make_array(std::move(width)),
        make_array(std::move(percent)),
    };
}

template <typename Array0, typename Array1, typename Array2>
EaseOfMovementBatchResult batch_ease_of_movement(
    EaseOfMovement &indicator,
    const Array0 &high,
    const Array1 &low,
    const Array2 &volume) {
    const std::size_t size = high.shape(0);
    require_same_size(size, low.shape(0));
    require_same_size(size, volume.shape(0));
    std::vector<double> ease(size);
    std::vector<double> sma(size);
    const auto *high_values = high.data();
    const auto *low_values = low.data();
    const auto *volume_values = volume.data();

    for (std::size_t i = 0; i < size; ++i) {
        const EaseOfMovementResult out = indicator.update(
            static_cast<double>(high_values[i]),
            static_cast<double>(low_values[i]),
            static_cast<double>(volume_values[i]));
        ease[i] = out.ease_of_movement;
        sma[i] = out.sma;
    }

    return {make_array(std::move(ease)), make_array(std::move(sma))};
}

template <typename Array>
KSTOscillatorBatchResult batch_kst(KSTOscillator &indicator, const Array &close) {
    const std::size_t size = close.shape(0);
    std::vector<double> kst(size);
    std::vector<double> signal(size);
    std::vector<double> difference(size);
    const auto *values = close.data();

    for (std::size_t i = 0; i < size; ++i) {
        const KSTOscillatorResult out = indicator.update(static_cast<double>(values[i]));
        kst[i] = out.kst;
        signal[i] = out.signal;
        difference[i] = out.difference;
    }

    return {make_array(std::move(kst)), make_array(std::move(signal)), make_array(std::move(difference))};
}

template <typename Array0, typename Array1>
IchimokuBatchResult batch_ichimoku(Ichimoku &indicator, const Array0 &high, const Array1 &low) {
    const std::size_t size = high.shape(0);
    require_same_size(size, low.shape(0));
    std::vector<double> conversion(size);
    std::vector<double> base(size);
    std::vector<double> span_a(size);
    std::vector<double> span_b(size);
    const auto *high_values = high.data();
    const auto *low_values = low.data();

    for (std::size_t i = 0; i < size; ++i) {
        const IchimokuResult out = indicator.update(static_cast<double>(high_values[i]), static_cast<double>(low_values[i]));
        conversion[i] = out.conversion;
        base[i] = out.base;
        span_a[i] = out.span_a;
        span_b[i] = out.span_b;
    }

    return {
        make_array(std::move(conversion)),
        make_array(std::move(base)),
        make_array(std::move(span_a)),
        make_array(std::move(span_b)),
    };
}

template <typename Array0, typename Array1>
FibonacciRetracementLevelsBatchResult batch_fibonacci_retracement_levels(
    FibonacciRetracementLevels &indicator,
    const Array0 &high,
    const Array1 &low) {
    return indicator.batch_array(high, low);
}

template <typename Array0, typename Array1, typename Array2>
VortexBatchResult batch_vortex(Vortex &indicator, const Array0 &close, const Array1 &high, const Array2 &low) {
    const std::size_t size = close.shape(0);
    require_same_size(size, high.shape(0));
    require_same_size(size, low.shape(0));
    std::vector<double> positive(size);
    std::vector<double> negative(size);
    std::vector<double> difference(size);
    const auto *close_values = close.data();
    const auto *high_values = high.data();
    const auto *low_values = low.data();

    for (std::size_t i = 0; i < size; ++i) {
        const VortexResult out = indicator.update(
            static_cast<double>(close_values[i]),
            static_cast<double>(high_values[i]),
            static_cast<double>(low_values[i]));
        positive[i] = out.positive;
        negative[i] = out.negative;
        difference[i] = out.difference;
    }

    return {make_array(std::move(positive)), make_array(std::move(negative)), make_array(std::move(difference))};
}

template <typename Indicator>
inline double call_update_checksum(Indicator &self, double arg0) {
    return result_checksum(self.update(arg0));
}

template <typename Indicator>
inline double call_update_checksum(Indicator &self, double arg0, double arg1) {
    return result_checksum(self.update(arg0, arg1));
}

template <typename Indicator>
inline double call_update_checksum(Indicator &self, double arg0, double arg1, double arg2) {
    return result_checksum(self.update(arg0, arg1, arg2));
}

template <typename Indicator>
inline double call_update_checksum(Indicator &self, double arg0, double arg1, double arg2, double arg3) {
    return result_checksum(self.update(arg0, arg1, arg2, arg3));
}

template <typename Indicator>
inline void call_advance_discard(Indicator &self, double arg0) {
    if constexpr (requires { self.advance(arg0); }) {
        self.advance(arg0);
    } else {
        (void) self.update(arg0);
    }
}

template <typename Indicator>
inline void call_advance_discard(Indicator &self, double arg0, double arg1) {
    if constexpr (requires { self.advance(arg0, arg1); }) {
        self.advance(arg0, arg1);
    } else {
        (void) self.update(arg0, arg1);
    }
}

template <typename Indicator>
inline void call_advance_discard(Indicator &self, double arg0, double arg1, double arg2) {
    if constexpr (requires { self.advance(arg0, arg1, arg2); }) {
        self.advance(arg0, arg1, arg2);
    } else {
        (void) self.update(arg0, arg1, arg2);
    }
}

template <typename Indicator>
inline void call_advance_discard(Indicator &self, double arg0, double arg1, double arg2, double arg3) {
    if constexpr (requires { self.advance(arg0, arg1, arg2, arg3); }) {
        self.advance(arg0, arg1, arg2, arg3);
    } else {
        (void) self.update(arg0, arg1, arg2, arg3);
    }
}

template <typename Indicator>
inline double call_advance_checksum(Indicator &self, double arg0) {
    if constexpr (requires { self.advance(arg0); }) {
        self.advance(arg0);
        if constexpr (requires { self.last(); }) {
            return result_checksum(self.last());
        }
        return 0.0;
    } else {
        return result_checksum(self.update(arg0));
    }
}

template <typename Indicator>
inline double call_advance_checksum(Indicator &self, double arg0, double arg1) {
    if constexpr (requires { self.advance(arg0, arg1); }) {
        self.advance(arg0, arg1);
        if constexpr (requires { self.last(); }) {
            return result_checksum(self.last());
        }
        return 0.0;
    } else {
        return result_checksum(self.update(arg0, arg1));
    }
}

template <typename Indicator>
inline double call_advance_checksum(Indicator &self, double arg0, double arg1, double arg2) {
    if constexpr (requires { self.advance(arg0, arg1, arg2); }) {
        self.advance(arg0, arg1, arg2);
        if constexpr (requires { self.last(); }) {
            return result_checksum(self.last());
        }
        return 0.0;
    } else {
        return result_checksum(self.update(arg0, arg1, arg2));
    }
}

template <typename Indicator>
inline double call_advance_checksum(Indicator &self, double arg0, double arg1, double arg2, double arg3) {
    if constexpr (requires { self.advance(arg0, arg1, arg2, arg3); }) {
        self.advance(arg0, arg1, arg2, arg3);
        if constexpr (requires { self.last(); }) {
            return result_checksum(self.last());
        }
        return 0.0;
    } else {
        return result_checksum(self.update(arg0, arg1, arg2, arg3));
    }
}

#define RTTA_ADVANCE1(TYPE, ARG0) \
    .def("advance", [](TYPE &self, double ARG0) { \
        call_advance_discard(self, ARG0); \
    }, nb::arg(#ARG0))

#define RTTA_ADVANCE2(TYPE, ARG0, ARG1) \
    .def("advance", [](TYPE &self, double ARG0, double ARG1) { \
        call_advance_discard(self, ARG0, ARG1); \
    }, nb::arg(#ARG0), nb::arg(#ARG1))

#define RTTA_ADVANCE3(TYPE, ARG0, ARG1, ARG2) \
    .def("advance", [](TYPE &self, double ARG0, double ARG1, double ARG2) { \
        call_advance_discard(self, ARG0, ARG1, ARG2); \
    }, nb::arg(#ARG0), nb::arg(#ARG1), nb::arg(#ARG2))

#define RTTA_ADVANCE4(TYPE, ARG0, ARG1, ARG2, ARG3) \
    .def("advance", [](TYPE &self, double ARG0, double ARG1, double ARG2, double ARG3) { \
        call_advance_discard(self, ARG0, ARG1, ARG2, ARG3); \
    }, nb::arg(#ARG0), nb::arg(#ARG1), nb::arg(#ARG2), nb::arg(#ARG3))

#define RTTA_REPLAY1(TYPE, ARG0) \
    .def("replay_update", [](TYPE &self, const InputArray &ARG0) { \
        const std::size_t size = ARG0.shape(0); \
        const auto *values0 = ARG0.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_update_checksum(self, static_cast<double>(values0[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0)) \
    .def("replay_update", [](TYPE &self, const FloatInputArray &ARG0) { \
        const std::size_t size = ARG0.shape(0); \
        const auto *values0 = ARG0.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_update_checksum(self, static_cast<double>(values0[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0)) \
    .def("replay_advance", [](TYPE &self, const InputArray &ARG0) { \
        const std::size_t size = ARG0.shape(0); \
        const auto *values0 = ARG0.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_advance_checksum(self, static_cast<double>(values0[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0)) \
    .def("replay_advance", [](TYPE &self, const FloatInputArray &ARG0) { \
        const std::size_t size = ARG0.shape(0); \
        const auto *values0 = ARG0.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_advance_checksum(self, static_cast<double>(values0[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0))

#define RTTA_REPLAY2(TYPE, ARG0, ARG1) \
    .def("replay_update", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_update_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1)) \
    .def("replay_update", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_update_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1)) \
    .def("replay_advance", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_advance_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1)) \
    .def("replay_advance", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_advance_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1))

#define RTTA_REPLAY3(TYPE, ARG0, ARG1, ARG2) \
    .def("replay_update", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1, const InputArray &ARG2) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        require_same_size(size, ARG2.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        const auto *values2 = ARG2.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_update_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i]), static_cast<double>(values2[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2)) \
    .def("replay_update", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1, const FloatInputArray &ARG2) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        require_same_size(size, ARG2.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        const auto *values2 = ARG2.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_update_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i]), static_cast<double>(values2[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2)) \
    .def("replay_advance", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1, const InputArray &ARG2) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        require_same_size(size, ARG2.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        const auto *values2 = ARG2.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_advance_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i]), static_cast<double>(values2[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2)) \
    .def("replay_advance", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1, const FloatInputArray &ARG2) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        require_same_size(size, ARG2.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        const auto *values2 = ARG2.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_advance_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i]), static_cast<double>(values2[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2))

#define RTTA_REPLAY4(TYPE, ARG0, ARG1, ARG2, ARG3) \
    .def("replay_update", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1, const InputArray &ARG2, const InputArray &ARG3) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        require_same_size(size, ARG2.shape(0)); \
        require_same_size(size, ARG3.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        const auto *values2 = ARG2.data(); \
        const auto *values3 = ARG3.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_update_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i]), static_cast<double>(values2[i]), static_cast<double>(values3[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2), array_arg(#ARG3)) \
    .def("replay_update", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1, const FloatInputArray &ARG2, const FloatInputArray &ARG3) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        require_same_size(size, ARG2.shape(0)); \
        require_same_size(size, ARG3.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        const auto *values2 = ARG2.data(); \
        const auto *values3 = ARG3.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_update_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i]), static_cast<double>(values2[i]), static_cast<double>(values3[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2), array_arg(#ARG3)) \
    .def("replay_advance", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1, const InputArray &ARG2, const InputArray &ARG3) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        require_same_size(size, ARG2.shape(0)); \
        require_same_size(size, ARG3.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        const auto *values2 = ARG2.data(); \
        const auto *values3 = ARG3.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_advance_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i]), static_cast<double>(values2[i]), static_cast<double>(values3[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2), array_arg(#ARG3)) \
    .def("replay_advance", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1, const FloatInputArray &ARG2, const FloatInputArray &ARG3) { \
        const std::size_t size = ARG0.shape(0); \
        require_same_size(size, ARG1.shape(0)); \
        require_same_size(size, ARG2.shape(0)); \
        require_same_size(size, ARG3.shape(0)); \
        const auto *values0 = ARG0.data(); \
        const auto *values1 = ARG1.data(); \
        const auto *values2 = ARG2.data(); \
        const auto *values3 = ARG3.data(); \
        double checksum = 0.0; \
        for (std::size_t i = 0; i < size; ++i) { \
            checksum += call_advance_checksum(self, static_cast<double>(values0[i]), static_cast<double>(values1[i]), static_cast<double>(values2[i]), static_cast<double>(values3[i])); \
        } \
        return checksum; \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2), array_arg(#ARG3))

#define RTTA_FIELD1(TYPE, FIELD, ARG0) \
    .def("update_" #FIELD, [](TYPE &self, double ARG0) { \
        return self.update(ARG0).FIELD; \
    }, nb::arg(#ARG0)) \
    .def("last_" #FIELD, [](const TYPE &self) { \
        return self.last().FIELD; \
    })

#define RTTA_FIELD2(TYPE, FIELD, ARG0, ARG1) \
    .def("update_" #FIELD, [](TYPE &self, double ARG0, double ARG1) { \
        return self.update(ARG0, ARG1).FIELD; \
    }, nb::arg(#ARG0), nb::arg(#ARG1)) \
    .def("last_" #FIELD, [](const TYPE &self) { \
        return self.last().FIELD; \
    })

#define RTTA_FIELD3(TYPE, FIELD, ARG0, ARG1, ARG2) \
    .def("update_" #FIELD, [](TYPE &self, double ARG0, double ARG1, double ARG2) { \
        return self.update(ARG0, ARG1, ARG2).FIELD; \
    }, nb::arg(#ARG0), nb::arg(#ARG1), nb::arg(#ARG2)) \
    .def("last_" #FIELD, [](const TYPE &self) { \
        return self.last().FIELD; \
    })

#define RTTA_FIELD4(TYPE, FIELD, ARG0, ARG1, ARG2, ARG3) \
    .def("update_" #FIELD, [](TYPE &self, double ARG0, double ARG1, double ARG2, double ARG3) { \
        return self.update(ARG0, ARG1, ARG2, ARG3).FIELD; \
    }, nb::arg(#ARG0), nb::arg(#ARG1), nb::arg(#ARG2), nb::arg(#ARG3)) \
    .def("last_" #FIELD, [](const TYPE &self) { \
        return self.last().FIELD; \
    })

#define RTTA_REPLAY_OUTPUTS1(TYPE, ARG0, BATCH_FUNC) \
    .def("replay_update_outputs", [](TYPE &self, const InputArray &ARG0) { \
        return BATCH_FUNC(self, ARG0); \
    }, array_arg(#ARG0)) \
    .def("replay_update_outputs", [](TYPE &self, const FloatInputArray &ARG0) { \
        return BATCH_FUNC(self, ARG0); \
    }, array_arg(#ARG0))

#define RTTA_REPLAY_OUTPUTS2(TYPE, ARG0, ARG1, BATCH_FUNC) \
    .def("replay_update_outputs", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1) { \
        return BATCH_FUNC(self, ARG0, ARG1); \
    }, array_arg(#ARG0), array_arg(#ARG1)) \
    .def("replay_update_outputs", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1) { \
        return BATCH_FUNC(self, ARG0, ARG1); \
    }, array_arg(#ARG0), array_arg(#ARG1))

#define RTTA_REPLAY_OUTPUTS3(TYPE, ARG0, ARG1, ARG2, BATCH_FUNC) \
    .def("replay_update_outputs", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1, const InputArray &ARG2) { \
        return BATCH_FUNC(self, ARG0, ARG1, ARG2); \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2)) \
    .def("replay_update_outputs", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1, const FloatInputArray &ARG2) { \
        return BATCH_FUNC(self, ARG0, ARG1, ARG2); \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2))

#define RTTA_REPLAY_OUTPUTS4(TYPE, ARG0, ARG1, ARG2, ARG3, BATCH_FUNC) \
    .def("replay_update_outputs", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1, const InputArray &ARG2, const InputArray &ARG3) { \
        return BATCH_FUNC(self, ARG0, ARG1, ARG2, ARG3); \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2), array_arg(#ARG3)) \
    .def("replay_update_outputs", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1, const FloatInputArray &ARG2, const FloatInputArray &ARG3) { \
        return BATCH_FUNC(self, ARG0, ARG1, ARG2, ARG3); \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2), array_arg(#ARG3))

#define RTTA_BATCH1(TYPE, ARG0, FIELD0) \
    .def("batch", [](TYPE &self, const InputArray &ARG0) { \
        return batch_update1(self, ARG0); \
    }, array_arg(#ARG0)) \
    .def("batch", [](TYPE &self, const FloatInputArray &ARG0) { \
        return batch_update1(self, ARG0); \
    }, array_arg(#ARG0)) \
    .def("batch", [](TYPE &self, nb::iterable records) { \
        return batch_records_one(self, records, FIELD0); \
    }, nb::arg("records"))

#define RTTA_BATCH2(TYPE, ARG0, ARG1, FIELD0, FIELD1) \
    .def("batch", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1) { \
        return batch_update2(self, ARG0, ARG1); \
    }, array_arg(#ARG0), array_arg(#ARG1)) \
    .def("batch", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1) { \
        return batch_update2(self, ARG0, ARG1); \
    }, array_arg(#ARG0), array_arg(#ARG1)) \
    .def("batch", [](TYPE &self, nb::iterable records) { \
        return batch_records_two(self, records, FIELD0, FIELD1); \
    }, nb::arg("records"))

#define RTTA_BATCH3(TYPE, ARG0, ARG1, ARG2, FIELD0, FIELD1, FIELD2) \
    .def("batch", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1, const InputArray &ARG2) { \
        return batch_update3(self, ARG0, ARG1, ARG2); \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2)) \
    .def("batch", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1, const FloatInputArray &ARG2) { \
        return batch_update3(self, ARG0, ARG1, ARG2); \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2)) \
    .def("batch", [](TYPE &self, nb::iterable records) { \
        return batch_records_three(self, records, FIELD0, FIELD1, FIELD2); \
    }, nb::arg("records"))

#define RTTA_BATCH4(TYPE, ARG0, ARG1, ARG2, ARG3, FIELD0, FIELD1, FIELD2, FIELD3) \
    .def("batch", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1, const InputArray &ARG2, const InputArray &ARG3) { \
        return batch_update4(self, ARG0, ARG1, ARG2, ARG3); \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2), array_arg(#ARG3)) \
    .def("batch", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1, const FloatInputArray &ARG2, const FloatInputArray &ARG3) { \
        return batch_update4(self, ARG0, ARG1, ARG2, ARG3); \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2), array_arg(#ARG3)) \
    .def("batch", [](TYPE &self, nb::iterable records) { \
        return batch_records_four(self, records, FIELD0, FIELD1, FIELD2, FIELD3); \
    }, nb::arg("records"))

#define RTTA_BATCH1_ARRAY(TYPE, ARG0, FIELD0) \
    .def("batch", [](TYPE &self, const InputArray &ARG0) { \
        return self.batch_array(ARG0); \
    }, array_arg(#ARG0)) \
    .def("batch", [](TYPE &self, const FloatInputArray &ARG0) { \
        return self.batch_array(ARG0); \
    }, array_arg(#ARG0)) \
    .def("batch", [](TYPE &self, nb::iterable records) { \
        if (table_has_column(records, FIELD0)) { \
            return dispatch_table1(self, records, FIELD0, [](auto &indicator, const auto &ARG0) { \
                return indicator.batch_array(ARG0); \
            }); \
        } \
        return batch_records_one(self, records, FIELD0); \
    }, nb::arg("records"))

#define RTTA_BATCH2_ARRAY(TYPE, ARG0, ARG1, FIELD0, FIELD1) \
    .def("batch", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1) { \
        return self.batch_array(ARG0, ARG1); \
    }, array_arg(#ARG0), array_arg(#ARG1)) \
    .def("batch", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1) { \
        return self.batch_array(ARG0, ARG1); \
    }, array_arg(#ARG0), array_arg(#ARG1)) \
    .def("batch", [](TYPE &self, nb::iterable records) { \
        if (table_has_column(records, FIELD0)) { \
            return dispatch_table2(self, records, FIELD0, FIELD1, [](auto &indicator, const auto &ARG0, const auto &ARG1) { \
                return indicator.batch_array(ARG0, ARG1); \
            }); \
        } \
        return batch_records_two(self, records, FIELD0, FIELD1); \
    }, nb::arg("records"))

#define RTTA_BATCH3_ARRAY(TYPE, ARG0, ARG1, ARG2, FIELD0, FIELD1, FIELD2) \
    .def("batch", [](TYPE &self, const InputArray &ARG0, const InputArray &ARG1, const InputArray &ARG2) { \
        return self.batch_array(ARG0, ARG1, ARG2); \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2)) \
    .def("batch", [](TYPE &self, const FloatInputArray &ARG0, const FloatInputArray &ARG1, const FloatInputArray &ARG2) { \
        return self.batch_array(ARG0, ARG1, ARG2); \
    }, array_arg(#ARG0), array_arg(#ARG1), array_arg(#ARG2)) \
    .def("batch", [](TYPE &self, nb::iterable records) { \
        if (table_has_column(records, FIELD0)) { \
            return dispatch_table3(self, records, FIELD0, FIELD1, FIELD2, [](auto &indicator, const auto &ARG0, const auto &ARG1, const auto &ARG2) { \
                return indicator.batch_array(ARG0, ARG1, ARG2); \
            }); \
        } \
        return batch_records_three(self, records, FIELD0, FIELD1, FIELD2); \
    }, nb::arg("records"))

}  // namespace

NB_MODULE(indicator, m) {
    m.doc() = "Low latency incremental technical analysis indicators.";

    nb::class_<EaseOfMovementResult>(m, "EaseOfMovementResult")
        .def_ro("ease_of_movement", &EaseOfMovementResult::ease_of_movement)
        .def_ro("sma", &EaseOfMovementResult::sma);

    nb::class_<EaseOfMovementBatchResult>(m, "EaseOfMovementBatchResult")
        .def_ro("ease_of_movement", &EaseOfMovementBatchResult::ease_of_movement)
        .def_ro("sma", &EaseOfMovementBatchResult::sma);

    nb::class_<LinearRegressionBatchResult>(m, "LinearRegressionBatchResult")
        .def_ro("value", &LinearRegressionBatchResult::value)
        .def_ro("slope", &LinearRegressionBatchResult::slope)
        .def_ro("intercept", &LinearRegressionBatchResult::intercept)
        .def_ro("angle", &LinearRegressionBatchResult::angle)
        .def_ro("tsf", &LinearRegressionBatchResult::tsf);

    nb::class_<HighLowResult>(m, "HighLowResult")
        .def_ro("min", &HighLowResult::min)
        .def_ro("max", &HighLowResult::max);

    nb::class_<HighLowBatchResult>(m, "HighLowBatchResult")
        .def_ro("min", &HighLowBatchResult::min)
        .def_ro("max", &HighLowBatchResult::max);

    nb::class_<HighLowIndexResult>(m, "HighLowIndexResult")
        .def_ro("min_index", &HighLowIndexResult::min_index)
        .def_ro("max_index", &HighLowIndexResult::max_index);

    nb::class_<HighLowIndexBatchResult>(m, "HighLowIndexBatchResult")
        .def_ro("min_index", &HighLowIndexBatchResult::min_index)
        .def_ro("max_index", &HighLowIndexBatchResult::max_index);

    nb::class_<KeltnerChannelResult>(m, "KeltnerChannelResult")
        .def_ro("middle", &KeltnerChannelResult::middle)
        .def_ro("upper", &KeltnerChannelResult::upper)
        .def_ro("lower", &KeltnerChannelResult::lower);

    nb::class_<KeltnerChannelBatchResult>(m, "KeltnerChannelBatchResult")
        .def_ro("middle", &KeltnerChannelBatchResult::middle)
        .def_ro("upper", &KeltnerChannelBatchResult::upper)
        .def_ro("lower", &KeltnerChannelBatchResult::lower);

    nb::class_<SuperTrendResult>(m, "SuperTrendResult")
        .def_ro("value", &SuperTrendResult::value)
        .def_ro("direction", &SuperTrendResult::direction)
        .def_ro("upper", &SuperTrendResult::upper)
        .def_ro("lower", &SuperTrendResult::lower);

    nb::class_<SuperTrendBatchResult>(m, "SuperTrendBatchResult")
        .def_ro("value", &SuperTrendBatchResult::value)
        .def_ro("direction", &SuperTrendBatchResult::direction)
        .def_ro("upper", &SuperTrendBatchResult::upper)
        .def_ro("lower", &SuperTrendBatchResult::lower);

    nb::class_<DonchianChannelResult>(m, "DonchianChannelResult")
        .def_ro("upper", &DonchianChannelResult::upper)
        .def_ro("lower", &DonchianChannelResult::lower)
        .def_ro("middle", &DonchianChannelResult::middle)
        .def_ro("width", &DonchianChannelResult::width)
        .def_ro("percent", &DonchianChannelResult::percent);

    nb::class_<DonchianChannelBatchResult>(m, "DonchianChannelBatchResult")
        .def_ro("upper", &DonchianChannelBatchResult::upper)
        .def_ro("lower", &DonchianChannelBatchResult::lower)
        .def_ro("middle", &DonchianChannelBatchResult::middle)
        .def_ro("width", &DonchianChannelBatchResult::width)
        .def_ro("percent", &DonchianChannelBatchResult::percent);

    nb::class_<FibonacciRetracementLevelsResult>(m, "FibonacciRetracementLevelsResult")
        .def_ro("level0", &FibonacciRetracementLevelsResult::level0)
        .def_ro("level236", &FibonacciRetracementLevelsResult::level236)
        .def_ro("level382", &FibonacciRetracementLevelsResult::level382)
        .def_ro("level500", &FibonacciRetracementLevelsResult::level500)
        .def_ro("level618", &FibonacciRetracementLevelsResult::level618)
        .def_ro("level100", &FibonacciRetracementLevelsResult::level100);

    nb::class_<FibonacciRetracementLevelsBatchResult>(m, "FibonacciRetracementLevelsBatchResult")
        .def_ro("level0", &FibonacciRetracementLevelsBatchResult::level0)
        .def_ro("level236", &FibonacciRetracementLevelsBatchResult::level236)
        .def_ro("level382", &FibonacciRetracementLevelsBatchResult::level382)
        .def_ro("level500", &FibonacciRetracementLevelsBatchResult::level500)
        .def_ro("level618", &FibonacciRetracementLevelsBatchResult::level618)
        .def_ro("level100", &FibonacciRetracementLevelsBatchResult::level100);

    nb::class_<AroonResult>(m, "AroonResult")
        .def_ro("down", &AroonResult::down)
        .def_ro("up", &AroonResult::up);

    nb::class_<AroonBatchResult>(m, "AroonBatchResult")
        .def_ro("down", &AroonBatchResult::down)
        .def_ro("up", &AroonBatchResult::up);

    nb::class_<VortexResult>(m, "VortexResult")
        .def_ro("positive", &VortexResult::positive)
        .def_ro("negative", &VortexResult::negative)
        .def_ro("difference", &VortexResult::difference);

    nb::class_<VortexBatchResult>(m, "VortexBatchResult")
        .def_ro("positive", &VortexBatchResult::positive)
        .def_ro("negative", &VortexBatchResult::negative)
        .def_ro("difference", &VortexBatchResult::difference);

    nb::class_<KSTOscillatorResult>(m, "KSTOscillatorResult")
        .def_ro("kst", &KSTOscillatorResult::kst)
        .def_ro("signal", &KSTOscillatorResult::signal)
        .def_ro("difference", &KSTOscillatorResult::difference);

    nb::class_<KSTOscillatorBatchResult>(m, "KSTOscillatorBatchResult")
        .def_ro("kst", &KSTOscillatorBatchResult::kst)
        .def_ro("signal", &KSTOscillatorBatchResult::signal)
        .def_ro("difference", &KSTOscillatorBatchResult::difference);

    nb::class_<IchimokuResult>(m, "IchimokuResult")
        .def_ro("conversion", &IchimokuResult::conversion)
        .def_ro("base", &IchimokuResult::base)
        .def_ro("span_a", &IchimokuResult::span_a)
        .def_ro("span_b", &IchimokuResult::span_b);

    nb::class_<IchimokuBatchResult>(m, "IchimokuBatchResult")
        .def_ro("conversion", &IchimokuBatchResult::conversion)
        .def_ro("base", &IchimokuBatchResult::base)
        .def_ro("span_a", &IchimokuBatchResult::span_a)
        .def_ro("span_b", &IchimokuBatchResult::span_b);

    nb::class_<PercentagePriceResult>(m, "PercentagePriceResult")
        .def_ro("ppo", &PercentagePriceResult::ppo)
        .def_ro("signal", &PercentagePriceResult::signal)
        .def_ro("histogram", &PercentagePriceResult::histogram);

    nb::class_<PercentagePriceBatchResult>(m, "PercentagePriceBatchResult")
        .def_ro("ppo", &PercentagePriceBatchResult::ppo)
        .def_ro("signal", &PercentagePriceBatchResult::signal)
        .def_ro("histogram", &PercentagePriceBatchResult::histogram);

    nb::class_<PercentageVolumeResult>(m, "PercentageVolumeResult")
        .def_ro("pvo", &PercentageVolumeResult::pvo)
        .def_ro("signal", &PercentageVolumeResult::signal)
        .def_ro("histogram", &PercentageVolumeResult::histogram);

    nb::class_<PercentageVolumeBatchResult>(m, "PercentageVolumeBatchResult")
        .def_ro("pvo", &PercentageVolumeBatchResult::pvo)
        .def_ro("signal", &PercentageVolumeBatchResult::signal)
        .def_ro("histogram", &PercentageVolumeBatchResult::histogram);

    nb::class_<FastStochasticResult>(m, "FastStochasticResult")
        .def_ro("fastk", &FastStochasticResult::fastk)
        .def_ro("fastd", &FastStochasticResult::fastd);

    nb::class_<FastStochasticBatchResult>(m, "FastStochasticBatchResult")
        .def_ro("fastk", &FastStochasticBatchResult::fastk)
        .def_ro("fastd", &FastStochasticBatchResult::fastd);

    nb::class_<StochasticResult>(m, "StochasticResult")
        .def_ro("slowk", &StochasticResult::slowk)
        .def_ro("slowd", &StochasticResult::slowd);

    nb::class_<StochasticBatchResult>(m, "StochasticBatchResult")
        .def_ro("slowk", &StochasticBatchResult::slowk)
        .def_ro("slowd", &StochasticBatchResult::slowd);

    nb::class_<BollingerBandsResult>(m, "BollingerBandsResult")
        .def_ro("middle", &BollingerBandsResult::middle)
        .def_ro("upper", &BollingerBandsResult::upper)
        .def_ro("lower", &BollingerBandsResult::lower);

    nb::class_<BollingerBandsBatchResult>(m, "BollingerBandsBatchResult")
        .def_ro("middle", &BollingerBandsBatchResult::middle)
        .def_ro("upper", &BollingerBandsBatchResult::upper)
        .def_ro("lower", &BollingerBandsBatchResult::lower);

    nb::class_<KalmanPredictionBandsResult>(m, "KalmanPredictionBandsResult")
        .def_ro("middle", &KalmanPredictionBandsResult::middle)
        .def_ro("upper", &KalmanPredictionBandsResult::upper)
        .def_ro("lower", &KalmanPredictionBandsResult::lower);

    nb::class_<KalmanPredictionBandsBatchResult>(m, "KalmanPredictionBandsBatchResult")
        .def_ro("middle", &KalmanPredictionBandsBatchResult::middle)
        .def_ro("upper", &KalmanPredictionBandsBatchResult::upper)
        .def_ro("lower", &KalmanPredictionBandsBatchResult::lower);

    nb::class_<KalmanTrendSignalResult>(m, "KalmanTrendSignalResult")
        .def_ro("trend", &KalmanTrendSignalResult::trend)
        .def_ro("signal", &KalmanTrendSignalResult::signal);

    nb::class_<KalmanTrendSignalBatchResult>(m, "KalmanTrendSignalBatchResult")
        .def_ro("trend", &KalmanTrendSignalBatchResult::trend)
        .def_ro("signal", &KalmanTrendSignalBatchResult::signal);

    nb::class_<KalmanLocalLinearTrendResult>(m, "KalmanLocalLinearTrendResult")
        .def_ro("level", &KalmanLocalLinearTrendResult::level)
        .def_ro("trend", &KalmanLocalLinearTrendResult::trend);

    nb::class_<KalmanLocalLinearTrendBatchResult>(m, "KalmanLocalLinearTrendBatchResult")
        .def_ro("level", &KalmanLocalLinearTrendBatchResult::level)
        .def_ro("trend", &KalmanLocalLinearTrendBatchResult::trend);

    nb::class_<RelativeVigorIndexResult>(m, "RelativeVigorIndexResult")
        .def_ro("rvi", &RelativeVigorIndexResult::rvi)
        .def_ro("signal", &RelativeVigorIndexResult::signal);

    nb::class_<RelativeVigorIndexBatchResult>(m, "RelativeVigorIndexBatchResult")
        .def_ro("rvi", &RelativeVigorIndexBatchResult::rvi)
        .def_ro("signal", &RelativeVigorIndexBatchResult::signal);

    nb::class_<KlingerVolumeOscillatorResult>(m, "KlingerVolumeOscillatorResult")
        .def_ro("kvo", &KlingerVolumeOscillatorResult::kvo)
        .def_ro("signal", &KlingerVolumeOscillatorResult::signal)
        .def_ro("histogram", &KlingerVolumeOscillatorResult::histogram);

    nb::class_<KlingerVolumeOscillatorBatchResult>(m, "KlingerVolumeOscillatorBatchResult")
        .def_ro("kvo", &KlingerVolumeOscillatorBatchResult::kvo)
        .def_ro("signal", &KlingerVolumeOscillatorBatchResult::signal)
        .def_ro("histogram", &KlingerVolumeOscillatorBatchResult::histogram);

    nb::class_<ElderRayIndexResult>(m, "ElderRayIndexResult")
        .def_ro("bull_power", &ElderRayIndexResult::bull_power)
        .def_ro("bear_power", &ElderRayIndexResult::bear_power);

    nb::class_<ElderRayIndexBatchResult>(m, "ElderRayIndexBatchResult")
        .def_ro("bull_power", &ElderRayIndexBatchResult::bull_power)
        .def_ro("bear_power", &ElderRayIndexBatchResult::bear_power);

    nb::class_<MesaAdaptiveMovingAverageResult>(m, "MesaAdaptiveMovingAverageResult")
        .def_ro("mama", &MesaAdaptiveMovingAverageResult::mama)
        .def_ro("fama", &MesaAdaptiveMovingAverageResult::fama);

    nb::class_<MesaAdaptiveMovingAverageBatchResult>(m, "MesaAdaptiveMovingAverageBatchResult")
        .def_ro("mama", &MesaAdaptiveMovingAverageBatchResult::mama)
        .def_ro("fama", &MesaAdaptiveMovingAverageBatchResult::fama);

    nb::class_<KalmanRegressionChannelResult>(m, "KalmanRegressionChannelResult")
        .def_ro("slope", &KalmanRegressionChannelResult::slope)
        .def_ro("intercept", &KalmanRegressionChannelResult::intercept)
        .def_ro("middle", &KalmanRegressionChannelResult::middle)
        .def_ro("upper", &KalmanRegressionChannelResult::upper)
        .def_ro("lower", &KalmanRegressionChannelResult::lower)
        .def_ro("spread", &KalmanRegressionChannelResult::spread);

    nb::class_<KalmanRegressionChannelBatchResult>(m, "KalmanRegressionChannelBatchResult")
        .def_ro("slope", &KalmanRegressionChannelBatchResult::slope)
        .def_ro("intercept", &KalmanRegressionChannelBatchResult::intercept)
        .def_ro("middle", &KalmanRegressionChannelBatchResult::middle)
        .def_ro("upper", &KalmanRegressionChannelBatchResult::upper)
        .def_ro("lower", &KalmanRegressionChannelBatchResult::lower)
        .def_ro("spread", &KalmanRegressionChannelBatchResult::spread);

    nb::class_<KalmanHedgeRatioResult>(m, "KalmanHedgeRatioResult")
        .def_ro("hedge_ratio", &KalmanHedgeRatioResult::hedge_ratio)
        .def_ro("intercept", &KalmanHedgeRatioResult::intercept)
        .def_ro("spread", &KalmanHedgeRatioResult::spread);

    nb::class_<KalmanHedgeRatioBatchResult>(m, "KalmanHedgeRatioBatchResult")
        .def_ro("hedge_ratio", &KalmanHedgeRatioBatchResult::hedge_ratio)
        .def_ro("intercept", &KalmanHedgeRatioBatchResult::intercept)
        .def_ro("spread", &KalmanHedgeRatioBatchResult::spread);

    nb::class_<TwoFactorKalmanTrendFilterResult>(m, "TwoFactorKalmanTrendFilterResult")
        .def_ro("short_trend", &TwoFactorKalmanTrendFilterResult::short_trend)
        .def_ro("long_trend", &TwoFactorKalmanTrendFilterResult::long_trend)
        .def_ro("value", &TwoFactorKalmanTrendFilterResult::value);

    nb::class_<TwoFactorKalmanTrendFilterBatchResult>(m, "TwoFactorKalmanTrendFilterBatchResult")
        .def_ro("short_trend", &TwoFactorKalmanTrendFilterBatchResult::short_trend)
        .def_ro("long_trend", &TwoFactorKalmanTrendFilterBatchResult::long_trend)
        .def_ro("value", &TwoFactorKalmanTrendFilterBatchResult::value);

    nb::class_<KalmanExtremumTrendResult>(m, "KalmanExtremumTrendResult")
        .def_ro("trend", &KalmanExtremumTrendResult::trend)
        .def_ro("oscillator", &KalmanExtremumTrendResult::oscillator)
        .def_ro("signal", &KalmanExtremumTrendResult::signal);

    nb::class_<KalmanExtremumTrendBatchResult>(m, "KalmanExtremumTrendBatchResult")
        .def_ro("trend", &KalmanExtremumTrendBatchResult::trend)
        .def_ro("oscillator", &KalmanExtremumTrendBatchResult::oscillator)
        .def_ro("signal", &KalmanExtremumTrendBatchResult::signal);

    nb::class_<AlphaBetaGammaTrackingFilterResult>(m, "AlphaBetaGammaTrackingFilterResult")
        .def_ro("price", &AlphaBetaGammaTrackingFilterResult::price)
        .def_ro("velocity", &AlphaBetaGammaTrackingFilterResult::velocity)
        .def_ro("acceleration", &AlphaBetaGammaTrackingFilterResult::acceleration)
        .def_ro("residual", &AlphaBetaGammaTrackingFilterResult::residual);

    nb::class_<AlphaBetaGammaTrackingFilterBatchResult>(m, "AlphaBetaGammaTrackingFilterBatchResult")
        .def_ro("price", &AlphaBetaGammaTrackingFilterBatchResult::price)
        .def_ro("velocity", &AlphaBetaGammaTrackingFilterBatchResult::velocity)
        .def_ro("acceleration", &AlphaBetaGammaTrackingFilterBatchResult::acceleration)
        .def_ro("residual", &AlphaBetaGammaTrackingFilterBatchResult::residual);

    nb::class_<InteractingMultipleModelFilterResult>(m, "InteractingMultipleModelFilterResult")
        .def_ro("value", &InteractingMultipleModelFilterResult::value)
        .def_ro("velocity", &InteractingMultipleModelFilterResult::velocity)
        .def_ro("low_vol_probability", &InteractingMultipleModelFilterResult::low_vol_probability)
        .def_ro("high_vol_probability", &InteractingMultipleModelFilterResult::high_vol_probability)
        .def_ro("trend_probability", &InteractingMultipleModelFilterResult::trend_probability)
        .def_ro("chop_probability", &InteractingMultipleModelFilterResult::chop_probability);

    nb::class_<InteractingMultipleModelFilterBatchResult>(m, "InteractingMultipleModelFilterBatchResult")
        .def_ro("value", &InteractingMultipleModelFilterBatchResult::value)
        .def_ro("velocity", &InteractingMultipleModelFilterBatchResult::velocity)
        .def_ro("low_vol_probability", &InteractingMultipleModelFilterBatchResult::low_vol_probability)
        .def_ro("high_vol_probability", &InteractingMultipleModelFilterBatchResult::high_vol_probability)
        .def_ro("trend_probability", &InteractingMultipleModelFilterBatchResult::trend_probability)
        .def_ro("chop_probability", &InteractingMultipleModelFilterBatchResult::chop_probability);

    nb::class_<ParticleFilterTrendResult>(m, "ParticleFilterTrendResult")
        .def_ro("trend", &ParticleFilterTrendResult::trend)
        .def_ro("velocity", &ParticleFilterTrendResult::velocity)
        .def_ro("signal", &ParticleFilterTrendResult::signal)
        .def_ro("effective_sample_size", &ParticleFilterTrendResult::effective_sample_size);

    nb::class_<ParticleFilterTrendBatchResult>(m, "ParticleFilterTrendBatchResult")
        .def_ro("trend", &ParticleFilterTrendBatchResult::trend)
        .def_ro("velocity", &ParticleFilterTrendBatchResult::velocity)
        .def_ro("signal", &ParticleFilterTrendBatchResult::signal)
        .def_ro("effective_sample_size", &ParticleFilterTrendBatchResult::effective_sample_size);

    nb::class_<SavitzkyGolayFilterResult>(m, "SavitzkyGolayFilterResult")
        .def_ro("smooth", &SavitzkyGolayFilterResult::smooth)
        .def_ro("first_derivative", &SavitzkyGolayFilterResult::first_derivative)
        .def_ro("second_derivative", &SavitzkyGolayFilterResult::second_derivative);

    nb::class_<SavitzkyGolayFilterBatchResult>(m, "SavitzkyGolayFilterBatchResult")
        .def_ro("smooth", &SavitzkyGolayFilterBatchResult::smooth)
        .def_ro("first_derivative", &SavitzkyGolayFilterBatchResult::first_derivative)
        .def_ro("second_derivative", &SavitzkyGolayFilterBatchResult::second_derivative);

    nb::class_<NadarayaWatsonEnvelopeResult>(m, "NadarayaWatsonEnvelopeResult")
        .def_ro("middle", &NadarayaWatsonEnvelopeResult::middle)
        .def_ro("upper", &NadarayaWatsonEnvelopeResult::upper)
        .def_ro("lower", &NadarayaWatsonEnvelopeResult::lower);

    nb::class_<NadarayaWatsonEnvelopeBatchResult>(m, "NadarayaWatsonEnvelopeBatchResult")
        .def_ro("middle", &NadarayaWatsonEnvelopeBatchResult::middle)
        .def_ro("upper", &NadarayaWatsonEnvelopeBatchResult::upper)
        .def_ro("lower", &NadarayaWatsonEnvelopeBatchResult::lower);

    nb::class_<GaussianProcessRegressionBandsResult>(m, "GaussianProcessRegressionBandsResult")
        .def_ro("middle", &GaussianProcessRegressionBandsResult::middle)
        .def_ro("upper", &GaussianProcessRegressionBandsResult::upper)
        .def_ro("lower", &GaussianProcessRegressionBandsResult::lower);

    nb::class_<GaussianProcessRegressionBandsBatchResult>(m, "GaussianProcessRegressionBandsBatchResult")
        .def_ro("middle", &GaussianProcessRegressionBandsBatchResult::middle)
        .def_ro("upper", &GaussianProcessRegressionBandsBatchResult::upper)
        .def_ro("lower", &GaussianProcessRegressionBandsBatchResult::lower);

    nb::class_<ZigZagSwingDetectorResult>(m, "ZigZagSwingDetectorResult")
        .def_ro("value", &ZigZagSwingDetectorResult::value)
        .def_ro("direction", &ZigZagSwingDetectorResult::direction)
        .def_ro("pivot", &ZigZagSwingDetectorResult::pivot)
        .def_ro("pivot_index", &ZigZagSwingDetectorResult::pivot_index);

    nb::class_<ZigZagSwingDetectorBatchResult>(m, "ZigZagSwingDetectorBatchResult")
        .def_ro("value", &ZigZagSwingDetectorBatchResult::value)
        .def_ro("direction", &ZigZagSwingDetectorBatchResult::direction)
        .def_ro("pivot", &ZigZagSwingDetectorBatchResult::pivot)
        .def_ro("pivot_index", &ZigZagSwingDetectorBatchResult::pivot_index);

    nb::class_<RenkoBrickGeneratorResult>(m, "RenkoBrickGeneratorResult")
        .def_ro("brick_open", &RenkoBrickGeneratorResult::brick_open)
        .def_ro("brick_close", &RenkoBrickGeneratorResult::brick_close)
        .def_ro("direction", &RenkoBrickGeneratorResult::direction)
        .def_ro("bricks", &RenkoBrickGeneratorResult::bricks)
        .def_ro("reversal", &RenkoBrickGeneratorResult::reversal);

    nb::class_<RenkoBrickGeneratorBatchResult>(m, "RenkoBrickGeneratorBatchResult")
        .def_ro("brick_open", &RenkoBrickGeneratorBatchResult::brick_open)
        .def_ro("brick_close", &RenkoBrickGeneratorBatchResult::brick_close)
        .def_ro("direction", &RenkoBrickGeneratorBatchResult::direction)
        .def_ro("bricks", &RenkoBrickGeneratorBatchResult::bricks)
        .def_ro("reversal", &RenkoBrickGeneratorBatchResult::reversal);

    nb::class_<HeikinAshiTransformResult>(m, "HeikinAshiTransformResult")
        .def_ro("open", &HeikinAshiTransformResult::open)
        .def_ro("high", &HeikinAshiTransformResult::high)
        .def_ro("low", &HeikinAshiTransformResult::low)
        .def_ro("close", &HeikinAshiTransformResult::close);

    nb::class_<HeikinAshiTransformBatchResult>(m, "HeikinAshiTransformBatchResult")
        .def_ro("open", &HeikinAshiTransformBatchResult::open)
        .def_ro("high", &HeikinAshiTransformBatchResult::high)
        .def_ro("low", &HeikinAshiTransformBatchResult::low)
        .def_ro("close", &HeikinAshiTransformBatchResult::close);

    nb::class_<VolumeProfileResult>(m, "VolumeProfileResult")
        .def_ro("point_of_control", &VolumeProfileResult::point_of_control)
        .def_ro("value_area_high", &VolumeProfileResult::value_area_high)
        .def_ro("value_area_low", &VolumeProfileResult::value_area_low);

    nb::class_<VolumeProfileBatchResult>(m, "VolumeProfileBatchResult")
        .def_ro("point_of_control", &VolumeProfileBatchResult::point_of_control)
        .def_ro("value_area_high", &VolumeProfileBatchResult::value_area_high)
        .def_ro("value_area_low", &VolumeProfileBatchResult::value_area_low);

    nb::class_<SpreadFeaturesResult>(m, "SpreadFeaturesResult")
        .def_ro("quoted_spread", &SpreadFeaturesResult::quoted_spread)
        .def_ro("effective_spread", &SpreadFeaturesResult::effective_spread)
        .def_ro("realized_spread", &SpreadFeaturesResult::realized_spread);

    nb::class_<SpreadFeaturesBatchResult>(m, "SpreadFeaturesBatchResult")
        .def_ro("quoted_spread", &SpreadFeaturesBatchResult::quoted_spread)
        .def_ro("effective_spread", &SpreadFeaturesBatchResult::effective_spread)
        .def_ro("realized_spread", &SpreadFeaturesBatchResult::realized_spread);

    nb::class_<AccumulationDistribution>(m, "AccumulationDistribution")
        .def(nb::init<>())
        .def("update", &AccumulationDistribution::update, nb::arg("close"), nb::arg("high"), nb::arg("low"), nb::arg("volume"))
        RTTA_ADVANCE4(AccumulationDistribution, close, high, low, volume)
        RTTA_REPLAY4(AccumulationDistribution, close, high, low, volume)
        .def("batch", [](AccumulationDistribution &self, const InputArray &close, const InputArray &high, const InputArray &low, const InputArray &volume) {
            return batch_update4(self, close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](AccumulationDistribution &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &volume) {
            return batch_update4(self, close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](AccumulationDistribution &self, nb::iterable records) {
            return batch_records_four(self, records, "close", "high", "low", "volume");
        }, nb::arg("records"));

    nb::class_<ChaikinOscillator>(m, "ChaikinOscillator")
        .def(nb::init<int, int>(), nb::arg("fast") = 3, nb::arg("slow") = 10)
        .def("update", &ChaikinOscillator::update, nb::arg("close"), nb::arg("high"), nb::arg("low"), nb::arg("volume"))
        RTTA_ADVANCE4(ChaikinOscillator, close, high, low, volume)
        RTTA_REPLAY4(ChaikinOscillator, close, high, low, volume)
        .def("batch", [](ChaikinOscillator &self, const InputArray &close, const InputArray &high, const InputArray &low, const InputArray &volume) {
            return batch_update4(self, close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](ChaikinOscillator &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &volume) {
            return batch_update4(self, close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](ChaikinOscillator &self, nb::iterable records) {
            return batch_records_four(self, records, "close", "high", "low", "volume");
        }, nb::arg("records"));

    nb::class_<AverageDirectionalMovementIndex>(m, "AverageDirectionalMovementIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &AverageDirectionalMovementIndex::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(AverageDirectionalMovementIndex, close, high, low)
        RTTA_REPLAY3(AverageDirectionalMovementIndex, close, high, low)
        RTTA_BATCH3_ARRAY(AverageDirectionalMovementIndex, close, high, low, "close", "high", "low");

    nb::class_<AverageDirectionalMovementIndexRating>(m, "AverageDirectionalMovementIndexRating")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &AverageDirectionalMovementIndexRating::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(AverageDirectionalMovementIndexRating, close, high, low)
        RTTA_REPLAY3(AverageDirectionalMovementIndexRating, close, high, low)
        .def("batch", [](AverageDirectionalMovementIndexRating &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return batch_update3(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](AverageDirectionalMovementIndexRating &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return batch_update3(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](AverageDirectionalMovementIndexRating &self, nb::iterable records) {
            return batch_records_three(self, records, "close", "high", "low");
        }, nb::arg("records"));

    nb::class_<AbsolutePriceOscillator>(m, "AbsolutePriceOscillator")
        .def(nb::init<double, double>(), nb::arg("fast") = 12.0, nb::arg("slow") = 26.0)
        .def("update", &AbsolutePriceOscillator::update, nb::arg("close"))
        RTTA_ADVANCE1(AbsolutePriceOscillator, close)
        RTTA_REPLAY1(AbsolutePriceOscillator, close)
        .def("batch", [](AbsolutePriceOscillator &self, const InputArray &close) {
            return batch_update1(self, close);
        }, array_arg("close"))
        .def("batch", [](AbsolutePriceOscillator &self, const FloatInputArray &close) {
            return batch_update1(self, close);
        }, array_arg("close"))
        .def("batch", [](AbsolutePriceOscillator &self, nb::iterable records) {
            return batch_records_one(self, records, "close");
        }, nb::arg("records"));

    nb::class_<Aroon>(m, "Aroon")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &Aroon::update, nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE2(Aroon, high, low)
        RTTA_REPLAY2(Aroon, high, low)
        RTTA_FIELD2(Aroon, down, high, low)
        RTTA_FIELD2(Aroon, up, high, low)
        RTTA_REPLAY_OUTPUTS2(Aroon, high, low, batch_aroon)
        .def("batch", [](Aroon &self, const InputArray &high, const InputArray &low) {
            return batch_aroon(self, high, low);
        }, array_arg("high"), array_arg("low"))
        .def("batch", [](Aroon &self, const FloatInputArray &high, const FloatInputArray &low) {
            return batch_aroon(self, high, low);
        }, array_arg("high"), array_arg("low"))
        .def("batch", [](Aroon &self, nb::iterable records) {
            if (table_has_column(records, "high")) {
                return dispatch_table2(self, records, "high", "low", [](auto &indicator, const auto &high, const auto &low) {
                    return batch_aroon(indicator, high, low);
                });
            }

            std::vector<double> down = make_record_output(records);
            std::vector<double> up;
            up.reserve(down.capacity());
            for (nb::handle record : records) {
                const AroonResult out = self.update(record_value(record, "high", 0), record_value(record, "low", 1));
                down.push_back(out.down);
                up.push_back(out.up);
            }
            return AroonBatchResult{make_array(std::move(down)), make_array(std::move(up))};
        }, nb::arg("records"));

    nb::class_<AroonOscillator>(m, "AroonOscillator")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &AroonOscillator::update, nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE2(AroonOscillator, high, low)
        RTTA_REPLAY2(AroonOscillator, high, low)
        .def("batch", [](AroonOscillator &self, const InputArray &high, const InputArray &low) {
            return self.batch_array(high, low);
        }, array_arg("high"), array_arg("low"))
        .def("batch", [](AroonOscillator &self, const FloatInputArray &high, const FloatInputArray &low) {
            return self.batch_array(high, low);
        }, array_arg("high"), array_arg("low"))
        .def("batch", [](AroonOscillator &self, nb::iterable records) {
            if (table_has_column(records, "high")) {
                return dispatch_table2(self, records, "high", "low", [](auto &indicator, const auto &high, const auto &low) {
                    return indicator.batch_array(high, low);
                });
            }
            return batch_records_two(self, records, "high", "low");
        }, nb::arg("records"));

    nb::class_<ATR>(m, "ATR")
        .def(nb::init<double, bool>(), nb::arg("window") = 14.0, nb::arg("fillna") = true)
        .def("update", &ATR::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(ATR, close, high, low)
        RTTA_REPLAY3(ATR, close, high, low)
        RTTA_BATCH3_ARRAY(ATR, close, high, low, "close", "high", "low");

    nb::class_<ATRP>(m, "ATRP")
        .def(nb::init<double, bool>(), nb::arg("window") = 14.0, nb::arg("fillna") = true)
        .def("update", &ATRP::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(ATRP, close, high, low)
        RTTA_REPLAY3(ATRP, close, high, low)
        RTTA_BATCH3_ARRAY(ATRP, close, high, low, "close", "high", "low");

    nb::class_<SuperTrend>(m, "SuperTrend")
        .def(nb::init<int, double, bool>(), nb::arg("window") = 10, nb::arg("multiplier") = 3.0, nb::arg("fillna") = true)
        .def("update", &SuperTrend::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(SuperTrend, close, high, low)
        RTTA_REPLAY3(SuperTrend, close, high, low)
        RTTA_FIELD3(SuperTrend, value, close, high, low)
        RTTA_FIELD3(SuperTrend, direction, close, high, low)
        RTTA_FIELD3(SuperTrend, upper, close, high, low)
        RTTA_FIELD3(SuperTrend, lower, close, high, low)
        RTTA_REPLAY_OUTPUTS3(SuperTrend, close, high, low, batch_super_trend)
        .def("batch", [](SuperTrend &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return batch_super_trend(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](SuperTrend &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return batch_super_trend(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](SuperTrend &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return batch_super_trend(indicator, close, high, low);
                });
            }

            std::vector<double> value = make_record_output(records);
            std::vector<double> direction;
            std::vector<double> upper;
            std::vector<double> lower;
            direction.reserve(value.capacity());
            upper.reserve(value.capacity());
            lower.reserve(value.capacity());
            for (nb::handle record : records) {
                const SuperTrendResult out = self.update(
                    record_value(record, "close", 0),
                    record_value(record, "high", 1),
                    record_value(record, "low", 2));
                value.push_back(out.value);
                direction.push_back(out.direction);
                upper.push_back(out.upper);
                lower.push_back(out.lower);
            }
            return SuperTrendBatchResult{
                make_array(std::move(value)),
                make_array(std::move(direction)),
                make_array(std::move(upper)),
                make_array(std::move(lower)),
            };
        }, nb::arg("records"));

    nb::class_<AveragePrice>(m, "AveragePrice")
        .def(nb::init<>())
        .def("update", &AveragePrice::update, nb::arg("open"), nb::arg("high"), nb::arg("low"), nb::arg("close"))
        RTTA_ADVANCE4(AveragePrice, open, high, low, close)
        RTTA_REPLAY4(AveragePrice, open, high, low, close)
        .def("batch", [](AveragePrice &self, const InputArray &open, const InputArray &high, const InputArray &low, const InputArray &close) {
            return batch_update4(self, open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("batch", [](AveragePrice &self, const FloatInputArray &open, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &close) {
            return batch_update4(self, open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("batch", [](AveragePrice &self, nb::iterable records) {
            return batch_records_four(self, records, "open", "high", "low", "close");
        }, nb::arg("records"));

    nb::class_<AwesomeOscillator>(m, "AwesomeOscillator")
        .def(nb::init<int, int, bool>(), nb::arg("window_1") = 34, nb::arg("window_2") = 5, nb::arg("fillna") = true)
        .def("update", &AwesomeOscillator::update, nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE2(AwesomeOscillator, high, low)
        RTTA_REPLAY2(AwesomeOscillator, high, low)
        .def("batch", &AwesomeOscillator::batch, array_arg("high"), array_arg("low"))
        .def("batch", [](AwesomeOscillator &self, const FloatInputArray &high, const FloatInputArray &low) {
            return batch_update2(self, high, low);
        }, array_arg("high"), array_arg("low"))
        .def("batch", [](AwesomeOscillator &self, nb::iterable records) {
            return batch_records_two(self, records, "high", "low");
        }, nb::arg("records"));

    nb::class_<Beta>(m, "Beta")
        .def(nb::init<int, bool>(), nb::arg("window") = 5, nb::arg("fillna") = true)
        .def("update", &Beta::update, nb::arg("real0"), nb::arg("real1"))
        RTTA_ADVANCE2(Beta, real0, real1)
        RTTA_REPLAY2(Beta, real0, real1)
        RTTA_BATCH2_ARRAY(Beta, real0, real1, "real0", "real1");

    nb::class_<BalanceOfPower>(m, "BalanceOfPower")
        .def(nb::init<>())
        .def("update", &BalanceOfPower::update, nb::arg("open"), nb::arg("high"), nb::arg("low"), nb::arg("close"))
        RTTA_ADVANCE4(BalanceOfPower, open, high, low, close)
        RTTA_REPLAY4(BalanceOfPower, open, high, low, close)
        .def("batch", [](BalanceOfPower &self, const InputArray &open, const InputArray &high, const InputArray &low, const InputArray &close) {
            return batch_update4(self, open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("batch", [](BalanceOfPower &self, const FloatInputArray &open, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &close) {
            return batch_update4(self, open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("batch", [](BalanceOfPower &self, nb::iterable records) {
            return batch_records_four(self, records, "open", "high", "low", "close");
        }, nb::arg("records"));

    nb::class_<CommodityChannelIndex>(m, "CommodityChannelIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &CommodityChannelIndex::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(CommodityChannelIndex, close, high, low)
        RTTA_REPLAY3(CommodityChannelIndex, close, high, low)
        RTTA_BATCH3(CommodityChannelIndex, close, high, low, "close", "high", "low");

    nb::class_<ChandeMomentumOscillator>(m, "ChandeMomentumOscillator")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &ChandeMomentumOscillator::update, nb::arg("close"))
        RTTA_ADVANCE1(ChandeMomentumOscillator, close)
        RTTA_REPLAY1(ChandeMomentumOscillator, close)
        .def("batch", [](ChandeMomentumOscillator &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](ChandeMomentumOscillator &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](ChandeMomentumOscillator &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            return batch_records_one(self, records, "close");
        }, nb::arg("records"));

    nb::class_<ChoppinessIndex>(m, "ChoppinessIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &ChoppinessIndex::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(ChoppinessIndex, close, high, low)
        RTTA_REPLAY3(ChoppinessIndex, close, high, low)
        .def("batch", [](ChoppinessIndex &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](ChoppinessIndex &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](ChoppinessIndex &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return indicator.batch_array(close, high, low);
                });
            }
            return batch_records_three(self, records, "close", "high", "low");
        }, nb::arg("records"));

    nb::class_<ConnorsRSI>(m, "ConnorsRSI")
        .def(nb::init<int, int, int, bool>(),
             nb::arg("rsi_window") = 3,
             nb::arg("streak_rsi_window") = 2,
             nb::arg("rank_window") = 100,
             nb::arg("fillna") = true)
        .def("update", &ConnorsRSI::update, nb::arg("close"))
        .def("last", &ConnorsRSI::last)
        RTTA_ADVANCE1(ConnorsRSI, close)
        RTTA_REPLAY1(ConnorsRSI, close)
        RTTA_BATCH1_ARRAY(ConnorsRSI, close, "close");

    nb::class_<CoppockCurve>(m, "CoppockCurve")
        .def(nb::init<int, int, int, bool>(),
             nb::arg("wma_window") = 10,
             nb::arg("long_roc") = 14,
             nb::arg("short_roc") = 11,
             nb::arg("fillna") = true)
        .def("update", &CoppockCurve::update, nb::arg("close"))
        .def("last", &CoppockCurve::last)
        RTTA_ADVANCE1(CoppockCurve, close)
        RTTA_REPLAY1(CoppockCurve, close)
        RTTA_BATCH1_ARRAY(CoppockCurve, close, "close");

    nb::class_<ElderRayIndex>(m, "ElderRayIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 13, nb::arg("fillna") = true)
        .def("update", &ElderRayIndex::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        .def("last", &ElderRayIndex::last)
        RTTA_ADVANCE3(ElderRayIndex, close, high, low)
        RTTA_REPLAY3(ElderRayIndex, close, high, low)
        RTTA_FIELD3(ElderRayIndex, bull_power, close, high, low)
        RTTA_FIELD3(ElderRayIndex, bear_power, close, high, low)
        .def("replay_update_outputs", [](ElderRayIndex &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("replay_update_outputs", [](ElderRayIndex &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](ElderRayIndex &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](ElderRayIndex &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](ElderRayIndex &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return indicator.batch_array(close, high, low);
                });
            }
            std::vector<double> bull = make_record_output(records);
            std::vector<double> bear;
            bear.reserve(bull.capacity());
            for (nb::handle record : records) {
                const ElderRayIndexResult out = self.update(record_value(record, "close", 0), record_value(record, "high", 1), record_value(record, "low", 2));
                bull.push_back(out.bull_power);
                bear.push_back(out.bear_power);
            }
            return ElderRayIndexBatchResult{make_array(std::move(bull)), make_array(std::move(bear))};
        }, nb::arg("records"));

    nb::class_<FisherTransform>(m, "FisherTransform")
        .def(nb::init<int, bool>(), nb::arg("window") = 10, nb::arg("fillna") = true)
        .def("update", &FisherTransform::update, nb::arg("high"), nb::arg("low"))
        .def("last", &FisherTransform::last)
        RTTA_ADVANCE2(FisherTransform, high, low)
        RTTA_REPLAY2(FisherTransform, high, low)
        RTTA_BATCH2_ARRAY(FisherTransform, high, low, "high", "low");

    nb::class_<FractalAdaptiveMovingAverage>(m, "FractalAdaptiveMovingAverage")
        .def(nb::init<int, bool>(), nb::arg("window") = 16, nb::arg("fillna") = true)
        .def("update", &FractalAdaptiveMovingAverage::update, nb::arg("value"))
        .def("last", &FractalAdaptiveMovingAverage::last)
        RTTA_ADVANCE1(FractalAdaptiveMovingAverage, value)
        RTTA_REPLAY1(FractalAdaptiveMovingAverage, value)
        RTTA_BATCH1_ARRAY(FractalAdaptiveMovingAverage, value, "value");

    nb::class_<RelativeVigorIndex>(m, "RelativeVigorIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 10, nb::arg("fillna") = true)
        .def("update", &RelativeVigorIndex::update, nb::arg("open"), nb::arg("high"), nb::arg("low"), nb::arg("close"))
        .def("last", &RelativeVigorIndex::last)
        RTTA_ADVANCE4(RelativeVigorIndex, open, high, low, close)
        RTTA_REPLAY4(RelativeVigorIndex, open, high, low, close)
        RTTA_FIELD4(RelativeVigorIndex, rvi, open, high, low, close)
        RTTA_FIELD4(RelativeVigorIndex, signal, open, high, low, close)
        .def("replay_update_outputs", [](RelativeVigorIndex &self, const InputArray &open, const InputArray &high, const InputArray &low, const InputArray &close) {
            return self.batch_array(open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("replay_update_outputs", [](RelativeVigorIndex &self, const FloatInputArray &open, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &close) {
            return self.batch_array(open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("batch", [](RelativeVigorIndex &self, const InputArray &open, const InputArray &high, const InputArray &low, const InputArray &close) {
            return self.batch_array(open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("batch", [](RelativeVigorIndex &self, const FloatInputArray &open, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &close) {
            return self.batch_array(open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("batch", [](RelativeVigorIndex &self, nb::iterable records) {
            if (table_has_column(records, "open")) {
                return dispatch_table4(self, records, "open", "high", "low", "close", [](auto &indicator, const auto &open, const auto &high, const auto &low, const auto &close) {
                    return indicator.batch_array(open, high, low, close);
                });
            }
            std::vector<double> rvi = make_record_output(records);
            std::vector<double> signal;
            signal.reserve(rvi.capacity());
            for (nb::handle record : records) {
                const RelativeVigorIndexResult out = self.update(
                    record_value(record, "open", 0),
                    record_value(record, "high", 1),
                    record_value(record, "low", 2),
                    record_value(record, "close", 3));
                rvi.push_back(out.rvi);
                signal.push_back(out.signal);
            }
            return RelativeVigorIndexBatchResult{make_array(std::move(rvi)), make_array(std::move(signal))};
        }, nb::arg("records"));

    nb::class_<KlingerVolumeOscillator>(m, "KlingerVolumeOscillator")
        .def(nb::init<int, int, int, bool>(),
             nb::arg("fast") = 34,
             nb::arg("slow") = 55,
             nb::arg("signal_window") = 13,
             nb::arg("fillna") = true)
        .def("update", &KlingerVolumeOscillator::update, nb::arg("close"), nb::arg("high"), nb::arg("low"), nb::arg("volume"))
        .def("last", &KlingerVolumeOscillator::last)
        RTTA_ADVANCE4(KlingerVolumeOscillator, close, high, low, volume)
        RTTA_REPLAY4(KlingerVolumeOscillator, close, high, low, volume)
        RTTA_FIELD4(KlingerVolumeOscillator, kvo, close, high, low, volume)
        RTTA_FIELD4(KlingerVolumeOscillator, signal, close, high, low, volume)
        RTTA_FIELD4(KlingerVolumeOscillator, histogram, close, high, low, volume)
        .def("replay_update_outputs", [](KlingerVolumeOscillator &self, const InputArray &close, const InputArray &high, const InputArray &low, const InputArray &volume) {
            return self.batch_array(close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("replay_update_outputs", [](KlingerVolumeOscillator &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &volume) {
            return self.batch_array(close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](KlingerVolumeOscillator &self, const InputArray &close, const InputArray &high, const InputArray &low, const InputArray &volume) {
            return self.batch_array(close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](KlingerVolumeOscillator &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &volume) {
            return self.batch_array(close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](KlingerVolumeOscillator &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table4(self, records, "close", "high", "low", "volume", [](auto &indicator, const auto &close, const auto &high, const auto &low, const auto &volume) {
                    return indicator.batch_array(close, high, low, volume);
                });
            }
            std::vector<double> kvo = make_record_output(records);
            std::vector<double> signal;
            std::vector<double> histogram;
            signal.reserve(kvo.capacity());
            histogram.reserve(kvo.capacity());
            for (nb::handle record : records) {
                const KlingerVolumeOscillatorResult out = self.update(
                    record_value(record, "close", 0),
                    record_value(record, "high", 1),
                    record_value(record, "low", 2),
                    record_value(record, "volume", 3));
                kvo.push_back(out.kvo);
                signal.push_back(out.signal);
                histogram.push_back(out.histogram);
            }
            return KlingerVolumeOscillatorBatchResult{
                make_array(std::move(kvo)),
                make_array(std::move(signal)),
                make_array(std::move(histogram)),
            };
        }, nb::arg("records"));

    nb::class_<Correlation>(m, "Correlation")
        .def(nb::init<int, bool>(), nb::arg("window") = 30, nb::arg("fillna") = true)
        .def("update", &Correlation::update, nb::arg("real0"), nb::arg("real1"))
        RTTA_ADVANCE2(Correlation, real0, real1)
        RTTA_REPLAY2(Correlation, real0, real1)
        RTTA_BATCH2_ARRAY(Correlation, real0, real1, "real0", "real1");

    nb::class_<CumulativeReturn>(m, "CumulativeReturn")
        .def(nb::init<>())
        .def("update", &CumulativeReturn::update, nb::arg("close"))
        RTTA_ADVANCE1(CumulativeReturn, close)
        RTTA_REPLAY1(CumulativeReturn, close)
        .def("batch", &CumulativeReturn::batch, array_arg("close"))
        .def("batch", [](CumulativeReturn &self, const FloatInputArray &close) {
            return batch_update1(self, close);
        }, array_arg("close"))
        .def("batch", [](CumulativeReturn &self, nb::iterable records) {
            return batch_records_one(self, records, "close");
        }, nb::arg("records"));

    nb::class_<DailyLogReturn>(m, "DailyLogReturn")
        .def(nb::init<bool>(), nb::arg("fillna") = true)
        .def("update", &DailyLogReturn::update, nb::arg("close"))
        RTTA_ADVANCE1(DailyLogReturn, close)
        RTTA_REPLAY1(DailyLogReturn, close)
        .def("batch", &DailyLogReturn::batch, array_arg("close"))
        .def("batch", [](DailyLogReturn &self, const FloatInputArray &close) {
            return batch_update1(self, close);
        }, array_arg("close"))
        .def("batch", [](DailyLogReturn &self, nb::iterable records) {
            return batch_records_one(self, records, "close");
        }, nb::arg("records"));

    nb::class_<DailyReturn>(m, "DailyReturn")
        .def(nb::init<bool>(), nb::arg("fillna") = true)
        .def("update", &DailyReturn::update, nb::arg("close"))
        RTTA_ADVANCE1(DailyReturn, close)
        RTTA_REPLAY1(DailyReturn, close)
        .def("batch", &DailyReturn::batch, array_arg("close"))
        .def("batch", [](DailyReturn &self, const FloatInputArray &close) {
            return batch_update1(self, close);
        }, array_arg("close"))
        .def("batch", [](DailyReturn &self, nb::iterable records) {
            return batch_records_one(self, records, "close");
        }, nb::arg("records"));

    nb::class_<Delay>(m, "Delay")
        .def(nb::init<int, bool>(), nb::arg("window") = 1, nb::arg("fillna") = true)
        .def("update", &Delay::update, nb::arg("value"))
        RTTA_ADVANCE1(Delay, value)
        RTTA_REPLAY1(Delay, value)
        .def("peek", &Delay::peek)
        RTTA_BATCH1(Delay, value, "value");

    nb::class_<DetrendedPriceOscillator>(m, "DetrendedPriceOscillator")
        .def(nb::init<int, bool>(), nb::arg("window") = 20, nb::arg("fillna") = true)
        .def("update", &DetrendedPriceOscillator::update, nb::arg("close"))
        RTTA_ADVANCE1(DetrendedPriceOscillator, close)
        RTTA_REPLAY1(DetrendedPriceOscillator, close)
        .def("batch", &DetrendedPriceOscillator::batch, array_arg("close"))
        .def("batch", [](DetrendedPriceOscillator &self, const FloatInputArray &close) {
            return batch_update1(self, close);
        }, array_arg("close"))
        .def("batch", [](DetrendedPriceOscillator &self, nb::iterable records) {
            return batch_records_one(self, records, "close");
        }, nb::arg("records"));

    nb::class_<DoubleEMA>(m, "DoubleEMA")
        .def(nb::init<double, bool>(), nb::arg("window") = 30.0, nb::arg("fillna") = true)
        .def("update", &DoubleEMA::update, nb::arg("value"))
        RTTA_ADVANCE1(DoubleEMA, value)
        RTTA_REPLAY1(DoubleEMA, value)
        RTTA_BATCH1(DoubleEMA, value, "value");

    nb::class_<DonchianChannel>(m, "DonchianChannel")
        .def(nb::init<int, bool>(), nb::arg("window") = 20, nb::arg("fillna") = true)
        .def("update", &DonchianChannel::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(DonchianChannel, close, high, low)
        RTTA_REPLAY3(DonchianChannel, close, high, low)
        RTTA_FIELD3(DonchianChannel, upper, close, high, low)
        RTTA_FIELD3(DonchianChannel, lower, close, high, low)
        RTTA_FIELD3(DonchianChannel, middle, close, high, low)
        RTTA_FIELD3(DonchianChannel, width, close, high, low)
        RTTA_FIELD3(DonchianChannel, percent, close, high, low)
        RTTA_REPLAY_OUTPUTS3(DonchianChannel, close, high, low, batch_donchian)
        .def("batch", &DonchianChannel::batch, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](DonchianChannel &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return batch_donchian(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](DonchianChannel &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return batch_donchian(indicator, close, high, low);
                });
            }

            std::vector<double> upper = make_record_output(records);
            std::vector<double> lower;
            std::vector<double> middle;
            std::vector<double> width;
            std::vector<double> percent;
            lower.reserve(upper.capacity());
            middle.reserve(upper.capacity());
            width.reserve(upper.capacity());
            percent.reserve(upper.capacity());
            for (nb::handle record : records) {
                const DonchianChannelResult out = self.update(
                    record_value(record, "close", 0),
                    record_value(record, "high", 1),
                    record_value(record, "low", 2));
                upper.push_back(out.upper);
                lower.push_back(out.lower);
                middle.push_back(out.middle);
                width.push_back(out.width);
                percent.push_back(out.percent);
            }
            return DonchianChannelBatchResult{
                make_array(std::move(upper)),
                make_array(std::move(lower)),
                make_array(std::move(middle)),
                make_array(std::move(width)),
                make_array(std::move(percent)),
            };
        }, nb::arg("records"));

    nb::class_<DirectionalMovementIndex>(m, "DirectionalMovementIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &DirectionalMovementIndex::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(DirectionalMovementIndex, close, high, low)
        RTTA_REPLAY3(DirectionalMovementIndex, close, high, low)
        RTTA_BATCH3_ARRAY(DirectionalMovementIndex, close, high, low, "close", "high", "low");

    nb::class_<EMA>(m, "EMA")
        .def(nb::init<double, bool>(), nb::arg("window"), nb::arg("fillna") = false)
        .def("update", &EMA::update, nb::arg("value"))
        RTTA_ADVANCE1(EMA, value)
        RTTA_REPLAY1(EMA, value)
        .def("batch", &EMA::batch, array_arg("input"))
        .def("batch", [](EMA &self, const FloatInputArray &input) {
            return batch_update1(self, input);
        }, array_arg("input"))
        .def("batch", [](EMA &self, nb::iterable records) {
            return batch_records_one(self, records, "input");
        }, nb::arg("records"));

    nb::class_<EWMA>(m, "EWMA")
        .def(nb::init<nb::object, nb::object, nb::object, bool>(),
             nb::arg("alpha") = nb::none(),
             nb::arg("span") = nb::none(),
             nb::arg("com") = nb::none(),
             nb::arg("fillna") = false)
        .def("update", &EWMA::update, nb::arg("value"))
        RTTA_ADVANCE1(EWMA, value)
        RTTA_REPLAY1(EWMA, value)
        RTTA_BATCH1(EWMA, value, "value");

    nb::class_<EaseOfMovement>(m, "EaseOfMovement")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &EaseOfMovement::update, nb::arg("high"), nb::arg("low"), nb::arg("volume"))
        RTTA_ADVANCE3(EaseOfMovement, high, low, volume)
        RTTA_REPLAY3(EaseOfMovement, high, low, volume)
        RTTA_FIELD3(EaseOfMovement, ease_of_movement, high, low, volume)
        RTTA_FIELD3(EaseOfMovement, sma, high, low, volume)
        RTTA_REPLAY_OUTPUTS3(EaseOfMovement, high, low, volume, batch_ease_of_movement)
        .def("batch", &EaseOfMovement::batch, array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](EaseOfMovement &self, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &volume) {
            return batch_ease_of_movement(self, high, low, volume);
        }, array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](EaseOfMovement &self, nb::iterable records) {
            if (table_has_column(records, "high")) {
                return dispatch_table3(self, records, "high", "low", "volume", [](auto &indicator, const auto &high, const auto &low, const auto &volume) {
                    return batch_ease_of_movement(indicator, high, low, volume);
                });
            }

            std::vector<double> ease = make_record_output(records);
            std::vector<double> sma;
            sma.reserve(ease.capacity());
            for (nb::handle record : records) {
                const EaseOfMovementResult out = self.update(
                    record_value(record, "high", 0),
                    record_value(record, "low", 1),
                    record_value(record, "volume", 2));
                ease.push_back(out.ease_of_movement);
                sma.push_back(out.sma);
            }
            return EaseOfMovementBatchResult{make_array(std::move(ease)), make_array(std::move(sma))};
        }, nb::arg("records"));

    nb::class_<ForceIndex>(m, "ForceIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 13, nb::arg("fillna") = true)
        .def("update", &ForceIndex::update, nb::arg("close"), nb::arg("volume"))
        RTTA_ADVANCE2(ForceIndex, close, volume)
        RTTA_REPLAY2(ForceIndex, close, volume)
        .def("batch", &ForceIndex::batch, array_arg("close"), array_arg("volume"))
        .def("batch", [](ForceIndex &self, const FloatInputArray &close, const FloatInputArray &volume) {
            return batch_update2(self, close, volume);
        }, array_arg("close"), array_arg("volume"))
        .def("batch", [](ForceIndex &self, nb::iterable records) {
            return batch_records_two(self, records, "close", "volume");
        }, nb::arg("records"));

    nb::class_<FibonacciRetracementLevels>(m, "FibonacciRetracementLevels")
        .def(nb::init<int, bool, bool>(), nb::arg("window") = 30, nb::arg("uptrend") = true, nb::arg("fillna") = true)
        .def("update", &FibonacciRetracementLevels::update, nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE2(FibonacciRetracementLevels, high, low)
        RTTA_REPLAY2(FibonacciRetracementLevels, high, low)
        RTTA_FIELD2(FibonacciRetracementLevels, level0, high, low)
        RTTA_FIELD2(FibonacciRetracementLevels, level236, high, low)
        RTTA_FIELD2(FibonacciRetracementLevels, level382, high, low)
        RTTA_FIELD2(FibonacciRetracementLevels, level500, high, low)
        RTTA_FIELD2(FibonacciRetracementLevels, level618, high, low)
        RTTA_FIELD2(FibonacciRetracementLevels, level100, high, low)
        RTTA_REPLAY_OUTPUTS2(FibonacciRetracementLevels, high, low, batch_fibonacci_retracement_levels)
        .def("batch", [](FibonacciRetracementLevels &self, const InputArray &high, const InputArray &low) {
            return self.batch_array(high, low);
        }, array_arg("high"), array_arg("low"))
        .def("batch", [](FibonacciRetracementLevels &self, const FloatInputArray &high, const FloatInputArray &low) {
            return self.batch_array(high, low);
        }, array_arg("high"), array_arg("low"))
        .def("batch", [](FibonacciRetracementLevels &self, nb::iterable records) {
            if (table_has_column(records, "high")) {
                return dispatch_table2(self, records, "high", "low", [](auto &indicator, const auto &high, const auto &low) {
                    return indicator.batch_array(high, low);
                });
            }

            std::vector<double> level0 = make_record_output(records);
            std::vector<double> level236;
            std::vector<double> level382;
            std::vector<double> level500;
            std::vector<double> level618;
            std::vector<double> level100;
            level236.reserve(level0.capacity());
            level382.reserve(level0.capacity());
            level500.reserve(level0.capacity());
            level618.reserve(level0.capacity());
            level100.reserve(level0.capacity());
            for (nb::handle record : records) {
                const FibonacciRetracementLevelsResult out = self.update(record_value(record, "high", 0), record_value(record, "low", 1));
                level0.push_back(out.level0);
                level236.push_back(out.level236);
                level382.push_back(out.level382);
                level500.push_back(out.level500);
                level618.push_back(out.level618);
                level100.push_back(out.level100);
            }
            return FibonacciRetracementLevelsBatchResult{
                make_array(std::move(level0)),
                make_array(std::move(level236)),
                make_array(std::move(level382)),
                make_array(std::move(level500)),
                make_array(std::move(level618)),
                make_array(std::move(level100)),
            };
        }, nb::arg("records"));

    nb::class_<Ichimoku>(m, "Ichimoku")
        .def(nb::init<int, int, int, bool>(),
             nb::arg("window1") = 9,
             nb::arg("window2") = 26,
             nb::arg("window3") = 52,
             nb::arg("fillna") = true)
        .def("update", &Ichimoku::update, nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE2(Ichimoku, high, low)
        RTTA_REPLAY2(Ichimoku, high, low)
        RTTA_FIELD2(Ichimoku, conversion, high, low)
        RTTA_FIELD2(Ichimoku, base, high, low)
        RTTA_FIELD2(Ichimoku, span_a, high, low)
        RTTA_FIELD2(Ichimoku, span_b, high, low)
        RTTA_REPLAY_OUTPUTS2(Ichimoku, high, low, batch_ichimoku)
        .def("batch", &Ichimoku::batch, array_arg("high"), array_arg("low"))
        .def("batch", [](Ichimoku &self, const FloatInputArray &high, const FloatInputArray &low) {
            return batch_ichimoku(self, high, low);
        }, array_arg("high"), array_arg("low"))
        .def("batch", [](Ichimoku &self, nb::iterable records) {
            if (table_has_column(records, "high")) {
                return dispatch_table2(self, records, "high", "low", [](auto &indicator, const auto &high, const auto &low) {
                    return batch_ichimoku(indicator, high, low);
                });
            }

            std::vector<double> conversion = make_record_output(records);
            std::vector<double> base;
            std::vector<double> span_a;
            std::vector<double> span_b;
            base.reserve(conversion.capacity());
            span_a.reserve(conversion.capacity());
            span_b.reserve(conversion.capacity());
            for (nb::handle record : records) {
                const IchimokuResult out = self.update(record_value(record, "high", 0), record_value(record, "low", 1));
                conversion.push_back(out.conversion);
                base.push_back(out.base);
                span_a.push_back(out.span_a);
                span_b.push_back(out.span_b);
            }
            return IchimokuBatchResult{
                make_array(std::move(conversion)),
                make_array(std::move(base)),
                make_array(std::move(span_a)),
                make_array(std::move(span_b)),
            };
        }, nb::arg("records"));

    nb::class_<Kama>(m, "Kama")
        .def(nb::init<int, int, int, bool>(),
             nb::arg("window") = 10,
             nb::arg("fast_ema") = 2,
             nb::arg("slow_ema") = 30,
             nb::arg("fillna") = true)
        .def("update", &Kama::update, nb::arg("close"))
        RTTA_ADVANCE1(Kama, close)
        RTTA_REPLAY1(Kama, close)
        .def("batch", &Kama::batch, array_arg("input"))
        .def("batch", [](Kama &self, const FloatInputArray &input) {
            return batch_update1(self, input);
        }, array_arg("input"))
        .def("batch", [](Kama &self, nb::iterable records) {
            return batch_records_one(self, records, "input");
        }, nb::arg("records"));

    nb::class_<KalmanMovingAverageTuning>(m, "KalmanMovingAverageTuning")
        .def_ro("initial_price", &KalmanMovingAverageTuning::initial_price)
        .def_ro("initial_velocity", &KalmanMovingAverageTuning::initial_velocity)
        .def_ro("dt", &KalmanMovingAverageTuning::dt)
        .def_ro("position_variance", &KalmanMovingAverageTuning::position_variance)
        .def_ro("velocity_variance", &KalmanMovingAverageTuning::velocity_variance)
        .def_ro("process_position_variance", &KalmanMovingAverageTuning::process_position_variance)
        .def_ro("process_velocity_variance", &KalmanMovingAverageTuning::process_velocity_variance)
        .def_ro("measurement_variance", &KalmanMovingAverageTuning::measurement_variance)
        .def("__len__", [](const KalmanMovingAverageTuning &) { return 8; })
        .def("__iter__", [](const KalmanMovingAverageTuning &self) {
            nb::tuple values = nb::make_tuple(
                self.initial_price,
                self.initial_velocity,
                self.dt,
                self.position_variance,
                self.velocity_variance,
                self.process_position_variance,
                self.process_velocity_variance,
                self.measurement_variance);
            return values.attr("__iter__")();
        });

    nb::class_<KalmanMovingAverage>(m, "KalmanMovingAverage")
        .def(nb::init<double, double, double, double, double, double, double, double, bool>(),
             nb::arg("initial_price") = nan(),
             nb::arg("initial_velocity") = 0.0,
             nb::arg("dt") = 1.0,
             nb::arg("position_variance") = 1.0,
             nb::arg("velocity_variance") = 1.0,
             nb::arg("process_position_variance") = 1.0e-4,
             nb::arg("process_velocity_variance") = 1.0e-3,
             nb::arg("measurement_variance") = 0.25,
             nb::arg("fillna") = true)
        .def(nb::init<const KalmanMovingAverageTuning &, bool>(),
             nb::arg("tuning"),
             nb::arg("fillna") = true)
        .def_static("tune", [](const InputArray &close, double dt, double min_variance) {
            return KalmanMovingAverage::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](const FloatInputArray &close, double dt, double min_variance) {
            return KalmanMovingAverage::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](nb::iterable records, double dt, double min_variance) {
            if (table_has_column(records, "close")) {
                nb::object close = table_column_array(records, "close");
                switch (array_dtype(close)) {
                    case InputDType::Float32:
                        return KalmanMovingAverage::tune_array(nb::cast<FloatInputArray>(close), dt, min_variance);
                    case InputDType::Float64:
                        return KalmanMovingAverage::tune_array(nb::cast<InputArray>(close), dt, min_variance);
                }
                throw nb::type_error("unsupported pandas table column dtype");
            }
            return KalmanMovingAverage::tune_records(records, dt, min_variance);
        }, nb::arg("records"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def("update", &KalmanMovingAverage::update, nb::arg("close"))
        .def("last", &KalmanMovingAverage::last)
        RTTA_ADVANCE1(KalmanMovingAverage, close)
        RTTA_REPLAY1(KalmanMovingAverage, close)
        .def("batch", [](KalmanMovingAverage &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanMovingAverage &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanMovingAverage &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> output = make_record_output(records);
            for (nb::handle record : records) {
                output.push_back(self.update(scalar_or_record_value(record, "close", 0)));
            }
            return make_array(std::move(output));
        }, nb::arg("records"));

    nb::class_<KalmanInnovationZScoreTuning>(m, "KalmanInnovationZScoreTuning")
        .def_ro("initial_price", &KalmanInnovationZScoreTuning::initial_price)
        .def_ro("initial_velocity", &KalmanInnovationZScoreTuning::initial_velocity)
        .def_ro("dt", &KalmanInnovationZScoreTuning::dt)
        .def_ro("position_variance", &KalmanInnovationZScoreTuning::position_variance)
        .def_ro("velocity_variance", &KalmanInnovationZScoreTuning::velocity_variance)
        .def_ro("process_position_variance", &KalmanInnovationZScoreTuning::process_position_variance)
        .def_ro("process_velocity_variance", &KalmanInnovationZScoreTuning::process_velocity_variance)
        .def_ro("measurement_variance", &KalmanInnovationZScoreTuning::measurement_variance)
        .def("__len__", [](const KalmanInnovationZScoreTuning &) { return 8; })
        .def("__iter__", [](const KalmanInnovationZScoreTuning &self) {
            nb::tuple values = nb::make_tuple(
                self.initial_price,
                self.initial_velocity,
                self.dt,
                self.position_variance,
                self.velocity_variance,
                self.process_position_variance,
                self.process_velocity_variance,
                self.measurement_variance);
            return values.attr("__iter__")();
        });

    nb::class_<KalmanInnovationZScore>(m, "KalmanInnovationZScore")
        .def(nb::init<double, double, double, double, double, double, double, double, bool>(),
             nb::arg("initial_price") = nan(),
             nb::arg("initial_velocity") = 0.0,
             nb::arg("dt") = 1.0,
             nb::arg("position_variance") = 1.0,
             nb::arg("velocity_variance") = 1.0,
             nb::arg("process_position_variance") = 1.0e-4,
             nb::arg("process_velocity_variance") = 1.0e-3,
             nb::arg("measurement_variance") = 0.25,
             nb::arg("fillna") = true)
        .def(nb::init<const KalmanInnovationZScoreTuning &, bool>(),
             nb::arg("tuning"),
             nb::arg("fillna") = true)
        .def_static("tune", [](const InputArray &close, double dt, double min_variance) {
            return KalmanInnovationZScore::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](const FloatInputArray &close, double dt, double min_variance) {
            return KalmanInnovationZScore::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](nb::iterable records, double dt, double min_variance) {
            if (table_has_column(records, "close")) {
                nb::object close = table_column_array(records, "close");
                switch (array_dtype(close)) {
                    case InputDType::Float32:
                        return KalmanInnovationZScore::tune_array(nb::cast<FloatInputArray>(close), dt, min_variance);
                    case InputDType::Float64:
                        return KalmanInnovationZScore::tune_array(nb::cast<InputArray>(close), dt, min_variance);
                }
                throw nb::type_error("unsupported pandas table column dtype");
            }
            return KalmanInnovationZScore::tune_records(records, dt, min_variance);
        }, nb::arg("records"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def("update", &KalmanInnovationZScore::update, nb::arg("close"))
        .def("last", &KalmanInnovationZScore::last)
        RTTA_ADVANCE1(KalmanInnovationZScore, close)
        RTTA_REPLAY1(KalmanInnovationZScore, close)
        .def("batch", [](KalmanInnovationZScore &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanInnovationZScore &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanInnovationZScore &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> output = make_record_output(records);
            for (nb::handle record : records) {
                output.push_back(self.update(scalar_or_record_value(record, "close", 0)));
            }
            return make_array(std::move(output));
        }, nb::arg("records"));

    nb::class_<KalmanPredictionBandsTuning>(m, "KalmanPredictionBandsTuning")
        .def_ro("initial_price", &KalmanPredictionBandsTuning::initial_price)
        .def_ro("initial_velocity", &KalmanPredictionBandsTuning::initial_velocity)
        .def_ro("dt", &KalmanPredictionBandsTuning::dt)
        .def_ro("position_variance", &KalmanPredictionBandsTuning::position_variance)
        .def_ro("velocity_variance", &KalmanPredictionBandsTuning::velocity_variance)
        .def_ro("process_position_variance", &KalmanPredictionBandsTuning::process_position_variance)
        .def_ro("process_velocity_variance", &KalmanPredictionBandsTuning::process_velocity_variance)
        .def_ro("measurement_variance", &KalmanPredictionBandsTuning::measurement_variance)
        .def("__len__", [](const KalmanPredictionBandsTuning &) { return 8; })
        .def("__iter__", [](const KalmanPredictionBandsTuning &self) {
            nb::tuple values = nb::make_tuple(
                self.initial_price,
                self.initial_velocity,
                self.dt,
                self.position_variance,
                self.velocity_variance,
                self.process_position_variance,
                self.process_velocity_variance,
                self.measurement_variance);
            return values.attr("__iter__")();
        });

    nb::class_<KalmanPredictionBands>(m, "KalmanPredictionBands")
        .def(nb::init<double, double, double, double, double, double, double, double, double, bool>(),
             nb::arg("initial_price") = nan(),
             nb::arg("initial_velocity") = 0.0,
             nb::arg("dt") = 1.0,
             nb::arg("position_variance") = 1.0,
             nb::arg("velocity_variance") = 1.0,
             nb::arg("process_position_variance") = 1.0e-4,
             nb::arg("process_velocity_variance") = 1.0e-3,
             nb::arg("measurement_variance") = 0.25,
             nb::arg("multiplier") = 2.0,
             nb::arg("fillna") = true)
        .def(nb::init<const KalmanPredictionBandsTuning &, double, bool>(),
             nb::arg("tuning"),
             nb::arg("multiplier") = 2.0,
             nb::arg("fillna") = true)
        .def_static("tune", [](const InputArray &close, double dt, double min_variance) {
            return KalmanPredictionBands::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](const FloatInputArray &close, double dt, double min_variance) {
            return KalmanPredictionBands::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](nb::iterable records, double dt, double min_variance) {
            if (table_has_column(records, "close")) {
                nb::object close = table_column_array(records, "close");
                switch (array_dtype(close)) {
                    case InputDType::Float32:
                        return KalmanPredictionBands::tune_array(nb::cast<FloatInputArray>(close), dt, min_variance);
                    case InputDType::Float64:
                        return KalmanPredictionBands::tune_array(nb::cast<InputArray>(close), dt, min_variance);
                }
                throw nb::type_error("unsupported pandas table column dtype");
            }
            return KalmanPredictionBands::tune_records(records, dt, min_variance);
        }, nb::arg("records"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def("update", &KalmanPredictionBands::update, nb::arg("close"))
        .def("last", &KalmanPredictionBands::last)
        RTTA_ADVANCE1(KalmanPredictionBands, close)
        RTTA_REPLAY1(KalmanPredictionBands, close)
        RTTA_FIELD1(KalmanPredictionBands, middle, close)
        RTTA_FIELD1(KalmanPredictionBands, upper, close)
        RTTA_FIELD1(KalmanPredictionBands, lower, close)
        .def("replay_update_outputs", [](KalmanPredictionBands &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](KalmanPredictionBands &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanPredictionBands &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanPredictionBands &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanPredictionBands &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> middle = make_record_output(records);
            std::vector<double> upper;
            std::vector<double> lower;
            upper.reserve(middle.capacity());
            lower.reserve(middle.capacity());
            for (nb::handle record : records) {
                const KalmanPredictionBandsResult out = self.update(scalar_or_record_value(record, "close", 0));
                middle.push_back(out.middle);
                upper.push_back(out.upper);
                lower.push_back(out.lower);
            }
            return KalmanPredictionBandsBatchResult{
                make_array(std::move(middle)),
                make_array(std::move(upper)),
                make_array(std::move(lower)),
            };
        }, nb::arg("records"));

    nb::class_<KalmanTrendSignalTuning>(m, "KalmanTrendSignalTuning")
        .def_ro("initial_price", &KalmanTrendSignalTuning::initial_price)
        .def_ro("initial_velocity", &KalmanTrendSignalTuning::initial_velocity)
        .def_ro("dt", &KalmanTrendSignalTuning::dt)
        .def_ro("position_variance", &KalmanTrendSignalTuning::position_variance)
        .def_ro("velocity_variance", &KalmanTrendSignalTuning::velocity_variance)
        .def_ro("process_position_variance", &KalmanTrendSignalTuning::process_position_variance)
        .def_ro("process_velocity_variance", &KalmanTrendSignalTuning::process_velocity_variance)
        .def_ro("measurement_variance", &KalmanTrendSignalTuning::measurement_variance)
        .def("__len__", [](const KalmanTrendSignalTuning &) { return 8; })
        .def("__iter__", [](const KalmanTrendSignalTuning &self) {
            nb::tuple values = nb::make_tuple(
                self.initial_price,
                self.initial_velocity,
                self.dt,
                self.position_variance,
                self.velocity_variance,
                self.process_position_variance,
                self.process_velocity_variance,
                self.measurement_variance);
            return values.attr("__iter__")();
        });

    nb::class_<KalmanTrendSignal>(m, "KalmanTrendSignal")
        .def(nb::init<double, double, double, double, double, double, double, double, bool>(),
             nb::arg("initial_price") = nan(),
             nb::arg("initial_velocity") = 0.0,
             nb::arg("dt") = 1.0,
             nb::arg("position_variance") = 1.0,
             nb::arg("velocity_variance") = 1.0,
             nb::arg("process_position_variance") = 1.0e-4,
             nb::arg("process_velocity_variance") = 1.0e-3,
             nb::arg("measurement_variance") = 0.25,
             nb::arg("fillna") = true)
        .def(nb::init<const KalmanTrendSignalTuning &, bool>(),
             nb::arg("tuning"),
             nb::arg("fillna") = true)
        .def_static("tune", [](const InputArray &close, double dt, double min_variance) {
            return KalmanTrendSignal::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](const FloatInputArray &close, double dt, double min_variance) {
            return KalmanTrendSignal::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](nb::iterable records, double dt, double min_variance) {
            if (table_has_column(records, "close")) {
                nb::object close = table_column_array(records, "close");
                switch (array_dtype(close)) {
                    case InputDType::Float32:
                        return KalmanTrendSignal::tune_array(nb::cast<FloatInputArray>(close), dt, min_variance);
                    case InputDType::Float64:
                        return KalmanTrendSignal::tune_array(nb::cast<InputArray>(close), dt, min_variance);
                }
                throw nb::type_error("unsupported pandas table column dtype");
            }
            return KalmanTrendSignal::tune_records(records, dt, min_variance);
        }, nb::arg("records"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def("update", &KalmanTrendSignal::update, nb::arg("close"))
        .def("last", &KalmanTrendSignal::last)
        RTTA_ADVANCE1(KalmanTrendSignal, close)
        RTTA_REPLAY1(KalmanTrendSignal, close)
        RTTA_FIELD1(KalmanTrendSignal, trend, close)
        RTTA_FIELD1(KalmanTrendSignal, signal, close)
        .def("replay_update_outputs", [](KalmanTrendSignal &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](KalmanTrendSignal &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanTrendSignal &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanTrendSignal &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanTrendSignal &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> trend = make_record_output(records);
            std::vector<double> signal;
            signal.reserve(trend.capacity());
            for (nb::handle record : records) {
                const KalmanTrendSignalResult out = self.update(scalar_or_record_value(record, "close", 0));
                trend.push_back(out.trend);
                signal.push_back(out.signal);
            }
            return KalmanTrendSignalBatchResult{
                make_array(std::move(trend)),
                make_array(std::move(signal)),
            };
        }, nb::arg("records"));

    nb::class_<KalmanVelocityOscillatorTuning>(m, "KalmanVelocityOscillatorTuning")
        .def_ro("initial_price", &KalmanVelocityOscillatorTuning::initial_price)
        .def_ro("initial_velocity", &KalmanVelocityOscillatorTuning::initial_velocity)
        .def_ro("dt", &KalmanVelocityOscillatorTuning::dt)
        .def_ro("position_variance", &KalmanVelocityOscillatorTuning::position_variance)
        .def_ro("velocity_variance", &KalmanVelocityOscillatorTuning::velocity_variance)
        .def_ro("process_position_variance", &KalmanVelocityOscillatorTuning::process_position_variance)
        .def_ro("process_velocity_variance", &KalmanVelocityOscillatorTuning::process_velocity_variance)
        .def_ro("measurement_variance", &KalmanVelocityOscillatorTuning::measurement_variance)
        .def("__len__", [](const KalmanVelocityOscillatorTuning &) { return 8; })
        .def("__iter__", [](const KalmanVelocityOscillatorTuning &self) {
            nb::tuple values = nb::make_tuple(
                self.initial_price,
                self.initial_velocity,
                self.dt,
                self.position_variance,
                self.velocity_variance,
                self.process_position_variance,
                self.process_velocity_variance,
                self.measurement_variance);
            return values.attr("__iter__")();
        });

    nb::class_<KalmanVelocityOscillator>(m, "KalmanVelocityOscillator")
        .def(nb::init<double, double, double, double, double, double, double, double, bool>(),
             nb::arg("initial_price") = nan(),
             nb::arg("initial_velocity") = 0.0,
             nb::arg("dt") = 1.0,
             nb::arg("position_variance") = 1.0,
             nb::arg("velocity_variance") = 1.0,
             nb::arg("process_position_variance") = 1.0e-4,
             nb::arg("process_velocity_variance") = 1.0e-3,
             nb::arg("measurement_variance") = 0.25,
             nb::arg("fillna") = true)
        .def(nb::init<const KalmanVelocityOscillatorTuning &, bool>(),
             nb::arg("tuning"),
             nb::arg("fillna") = true)
        .def_static("tune", [](const InputArray &close, double dt, double min_variance) {
            return KalmanVelocityOscillator::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](const FloatInputArray &close, double dt, double min_variance) {
            return KalmanVelocityOscillator::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](nb::iterable records, double dt, double min_variance) {
            if (table_has_column(records, "close")) {
                nb::object close = table_column_array(records, "close");
                switch (array_dtype(close)) {
                    case InputDType::Float32:
                        return KalmanVelocityOscillator::tune_array(nb::cast<FloatInputArray>(close), dt, min_variance);
                    case InputDType::Float64:
                        return KalmanVelocityOscillator::tune_array(nb::cast<InputArray>(close), dt, min_variance);
                }
                throw nb::type_error("unsupported pandas table column dtype");
            }
            return KalmanVelocityOscillator::tune_records(records, dt, min_variance);
        }, nb::arg("records"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def("update", &KalmanVelocityOscillator::update, nb::arg("close"))
        .def("last", &KalmanVelocityOscillator::last)
        RTTA_ADVANCE1(KalmanVelocityOscillator, close)
        RTTA_REPLAY1(KalmanVelocityOscillator, close)
        .def("batch", [](KalmanVelocityOscillator &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanVelocityOscillator &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanVelocityOscillator &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> output = make_record_output(records);
            for (nb::handle record : records) {
                output.push_back(self.update(scalar_or_record_value(record, "close", 0)));
            }
            return make_array(std::move(output));
        }, nb::arg("records"));

    nb::class_<KalmanLocalLinearTrendTuning>(m, "KalmanLocalLinearTrendTuning")
        .def_ro("initial_level", &KalmanLocalLinearTrendTuning::initial_level)
        .def_ro("initial_trend", &KalmanLocalLinearTrendTuning::initial_trend)
        .def_ro("dt", &KalmanLocalLinearTrendTuning::dt)
        .def_ro("level_variance", &KalmanLocalLinearTrendTuning::level_variance)
        .def_ro("trend_variance", &KalmanLocalLinearTrendTuning::trend_variance)
        .def_ro("process_level_variance", &KalmanLocalLinearTrendTuning::process_level_variance)
        .def_ro("process_trend_variance", &KalmanLocalLinearTrendTuning::process_trend_variance)
        .def_ro("observation_variance", &KalmanLocalLinearTrendTuning::observation_variance)
        .def("__len__", [](const KalmanLocalLinearTrendTuning &) { return 8; })
        .def("__iter__", [](const KalmanLocalLinearTrendTuning &self) {
            nb::tuple values = nb::make_tuple(
                self.initial_level,
                self.initial_trend,
                self.dt,
                self.level_variance,
                self.trend_variance,
                self.process_level_variance,
                self.process_trend_variance,
                self.observation_variance);
            return values.attr("__iter__")();
        });

    nb::class_<KalmanLocalLinearTrend>(m, "KalmanLocalLinearTrend")
        .def(nb::init<double, double, double, double, double, double, double, double, bool>(),
             nb::arg("initial_level") = nan(),
             nb::arg("initial_trend") = 0.0,
             nb::arg("dt") = 1.0,
             nb::arg("level_variance") = 1.0,
             nb::arg("trend_variance") = 1.0,
             nb::arg("process_level_variance") = 1.0e-4,
             nb::arg("process_trend_variance") = 1.0e-3,
             nb::arg("observation_variance") = 0.25,
             nb::arg("fillna") = true)
        .def(nb::init<const KalmanLocalLinearTrendTuning &, bool>(),
             nb::arg("tuning"),
             nb::arg("fillna") = true)
        .def_static("tune", [](const InputArray &close, double dt, double min_variance) {
            return KalmanLocalLinearTrend::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](const FloatInputArray &close, double dt, double min_variance) {
            return KalmanLocalLinearTrend::tune_array(close, dt, min_variance);
        }, array_arg("close"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def_static("tune", [](nb::iterable records, double dt, double min_variance) {
            if (table_has_column(records, "close")) {
                nb::object close = table_column_array(records, "close");
                switch (array_dtype(close)) {
                    case InputDType::Float32:
                        return KalmanLocalLinearTrend::tune_array(nb::cast<FloatInputArray>(close), dt, min_variance);
                    case InputDType::Float64:
                        return KalmanLocalLinearTrend::tune_array(nb::cast<InputArray>(close), dt, min_variance);
                }
                throw nb::type_error("unsupported pandas table column dtype");
            }
            return KalmanLocalLinearTrend::tune_records(records, dt, min_variance);
        }, nb::arg("records"), nb::arg("dt") = 1.0, nb::arg("min_variance") = 1.0e-12)
        .def("update", &KalmanLocalLinearTrend::update, nb::arg("close"))
        .def("last", &KalmanLocalLinearTrend::last)
        RTTA_ADVANCE1(KalmanLocalLinearTrend, close)
        RTTA_REPLAY1(KalmanLocalLinearTrend, close)
        RTTA_FIELD1(KalmanLocalLinearTrend, level, close)
        RTTA_FIELD1(KalmanLocalLinearTrend, trend, close)
        .def("replay_update_outputs", [](KalmanLocalLinearTrend &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](KalmanLocalLinearTrend &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanLocalLinearTrend &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanLocalLinearTrend &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](KalmanLocalLinearTrend &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> level = make_record_output(records);
            std::vector<double> trend;
            trend.reserve(level.capacity());
            for (nb::handle record : records) {
                const KalmanLocalLinearTrendResult out = self.update(scalar_or_record_value(record, "close", 0));
                level.push_back(out.level);
                trend.push_back(out.trend);
            }
            return KalmanLocalLinearTrendBatchResult{
                make_array(std::move(level)),
                make_array(std::move(trend)),
            };
        }, nb::arg("records"));

    nb::class_<KalmanHedgeRatio>(m, "KalmanHedgeRatio")
        .def(nb::init<double, double, double, double, double, bool>(),
             nb::arg("initial_hedge_ratio") = 1.0,
             nb::arg("initial_intercept") = 0.0,
             nb::arg("initial_variance") = 1.0,
             nb::arg("process_variance") = 1.0e-5,
             nb::arg("measurement_variance") = 1.0,
             nb::arg("fillna") = true)
        .def("update", &KalmanHedgeRatio::update, nb::arg("real0"), nb::arg("real1"))
        .def("last", &KalmanHedgeRatio::last)
        RTTA_ADVANCE2(KalmanHedgeRatio, real0, real1)
        RTTA_REPLAY2(KalmanHedgeRatio, real0, real1)
        RTTA_FIELD2(KalmanHedgeRatio, hedge_ratio, real0, real1)
        RTTA_FIELD2(KalmanHedgeRatio, intercept, real0, real1)
        RTTA_FIELD2(KalmanHedgeRatio, spread, real0, real1)
        .def("replay_update_outputs", [](KalmanHedgeRatio &self, const InputArray &real0, const InputArray &real1) {
            return self.batch_array(real0, real1);
        }, array_arg("real0"), array_arg("real1"))
        .def("replay_update_outputs", [](KalmanHedgeRatio &self, const FloatInputArray &real0, const FloatInputArray &real1) {
            return self.batch_array(real0, real1);
        }, array_arg("real0"), array_arg("real1"))
        .def("batch", [](KalmanHedgeRatio &self, const InputArray &real0, const InputArray &real1) {
            return self.batch_array(real0, real1);
        }, array_arg("real0"), array_arg("real1"))
        .def("batch", [](KalmanHedgeRatio &self, const FloatInputArray &real0, const FloatInputArray &real1) {
            return self.batch_array(real0, real1);
        }, array_arg("real0"), array_arg("real1"))
        .def("batch", [](KalmanHedgeRatio &self, nb::iterable records) {
            if (table_has_column(records, "real0")) {
                return dispatch_table2(self, records, "real0", "real1", [](auto &indicator, const auto &real0, const auto &real1) {
                    return indicator.batch_array(real0, real1);
                });
            }
            std::vector<double> hedge = make_record_output(records);
            std::vector<double> intercept;
            std::vector<double> spread;
            intercept.reserve(hedge.capacity());
            spread.reserve(hedge.capacity());
            for (nb::handle record : records) {
                const KalmanHedgeRatioResult out = self.update(record_value(record, "real0", 0), record_value(record, "real1", 1));
                hedge.push_back(out.hedge_ratio);
                intercept.push_back(out.intercept);
                spread.push_back(out.spread);
            }
            return KalmanHedgeRatioBatchResult{
                make_array(std::move(hedge)),
                make_array(std::move(intercept)),
                make_array(std::move(spread)),
            };
        }, nb::arg("records"));

    nb::class_<KalmanRegressionChannel>(m, "KalmanRegressionChannel")
        .def(nb::init<double, double, double, double, double, double, bool>(),
             nb::arg("initial_slope") = 1.0,
             nb::arg("initial_intercept") = 0.0,
             nb::arg("initial_variance") = 1.0,
             nb::arg("process_variance") = 1.0e-5,
             nb::arg("measurement_variance") = 1.0,
             nb::arg("multiplier") = 2.0,
             nb::arg("fillna") = true)
        .def("update", &KalmanRegressionChannel::update, nb::arg("real0"), nb::arg("real1"))
        .def("last", &KalmanRegressionChannel::last)
        RTTA_ADVANCE2(KalmanRegressionChannel, real0, real1)
        RTTA_REPLAY2(KalmanRegressionChannel, real0, real1)
        RTTA_FIELD2(KalmanRegressionChannel, slope, real0, real1)
        RTTA_FIELD2(KalmanRegressionChannel, intercept, real0, real1)
        RTTA_FIELD2(KalmanRegressionChannel, middle, real0, real1)
        RTTA_FIELD2(KalmanRegressionChannel, upper, real0, real1)
        RTTA_FIELD2(KalmanRegressionChannel, lower, real0, real1)
        RTTA_FIELD2(KalmanRegressionChannel, spread, real0, real1)
        .def("replay_update_outputs", [](KalmanRegressionChannel &self, const InputArray &real0, const InputArray &real1) {
            return self.batch_array(real0, real1);
        }, array_arg("real0"), array_arg("real1"))
        .def("replay_update_outputs", [](KalmanRegressionChannel &self, const FloatInputArray &real0, const FloatInputArray &real1) {
            return self.batch_array(real0, real1);
        }, array_arg("real0"), array_arg("real1"))
        .def("batch", [](KalmanRegressionChannel &self, const InputArray &real0, const InputArray &real1) {
            return self.batch_array(real0, real1);
        }, array_arg("real0"), array_arg("real1"))
        .def("batch", [](KalmanRegressionChannel &self, const FloatInputArray &real0, const FloatInputArray &real1) {
            return self.batch_array(real0, real1);
        }, array_arg("real0"), array_arg("real1"))
        .def("batch", [](KalmanRegressionChannel &self, nb::iterable records) {
            if (table_has_column(records, "real0")) {
                return dispatch_table2(self, records, "real0", "real1", [](auto &indicator, const auto &real0, const auto &real1) {
                    return indicator.batch_array(real0, real1);
                });
            }
            std::vector<double> slope = make_record_output(records);
            std::vector<double> intercept;
            std::vector<double> middle;
            std::vector<double> upper;
            std::vector<double> lower;
            std::vector<double> spread;
            intercept.reserve(slope.capacity());
            middle.reserve(slope.capacity());
            upper.reserve(slope.capacity());
            lower.reserve(slope.capacity());
            spread.reserve(slope.capacity());
            for (nb::handle record : records) {
                const KalmanRegressionChannelResult out = self.update(record_value(record, "real0", 0), record_value(record, "real1", 1));
                slope.push_back(out.slope);
                intercept.push_back(out.intercept);
                middle.push_back(out.middle);
                upper.push_back(out.upper);
                lower.push_back(out.lower);
                spread.push_back(out.spread);
            }
            return KalmanRegressionChannelBatchResult{
                make_array(std::move(slope)),
                make_array(std::move(intercept)),
                make_array(std::move(middle)),
                make_array(std::move(upper)),
                make_array(std::move(lower)),
                make_array(std::move(spread)),
            };
        }, nb::arg("records"));

    nb::class_<TwoFactorKalmanTrendFilter>(m, "TwoFactorKalmanTrendFilter")
        .def(nb::init<double, double, double, double, double, double, double, double, bool>(),
             nb::arg("short_persistence") = 0.55,
             nb::arg("short_to_long") = 0.45,
             nb::arg("long_persistence") = 0.98,
             nb::arg("short_loading") = 1.0,
             nb::arg("long_loading") = 1.0,
             nb::arg("process_short_variance") = 1.0e-3,
             nb::arg("process_long_variance") = 1.0e-5,
             nb::arg("measurement_variance") = 0.25,
             nb::arg("fillna") = true)
        .def("update", &TwoFactorKalmanTrendFilter::update, nb::arg("close"))
        .def("last", &TwoFactorKalmanTrendFilter::last)
        RTTA_ADVANCE1(TwoFactorKalmanTrendFilter, close)
        RTTA_REPLAY1(TwoFactorKalmanTrendFilter, close)
        RTTA_FIELD1(TwoFactorKalmanTrendFilter, short_trend, close)
        RTTA_FIELD1(TwoFactorKalmanTrendFilter, long_trend, close)
        RTTA_FIELD1(TwoFactorKalmanTrendFilter, value, close)
        .def("replay_update_outputs", [](TwoFactorKalmanTrendFilter &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](TwoFactorKalmanTrendFilter &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](TwoFactorKalmanTrendFilter &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](TwoFactorKalmanTrendFilter &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](TwoFactorKalmanTrendFilter &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> short_trend = make_record_output(records);
            std::vector<double> long_trend;
            std::vector<double> value;
            long_trend.reserve(short_trend.capacity());
            value.reserve(short_trend.capacity());
            for (nb::handle record : records) {
                const TwoFactorKalmanTrendFilterResult out = self.update(scalar_or_record_value(record, "close", 0));
                short_trend.push_back(out.short_trend);
                long_trend.push_back(out.long_trend);
                value.push_back(out.value);
            }
            return TwoFactorKalmanTrendFilterBatchResult{
                make_array(std::move(short_trend)),
                make_array(std::move(long_trend)),
                make_array(std::move(value)),
            };
        }, nb::arg("records"));

    nb::class_<KalmanExtremumTrend>(m, "KalmanExtremumTrend")
        .def(nb::init<int, double, double, double, double, double, double, double, double, bool>(),
             nb::arg("window") = 14,
             nb::arg("initial_price") = nan(),
             nb::arg("initial_velocity") = 0.0,
             nb::arg("dt") = 1.0,
             nb::arg("position_variance") = 1.0,
             nb::arg("velocity_variance") = 1.0,
             nb::arg("process_position_variance") = 1.0e-4,
             nb::arg("process_velocity_variance") = 1.0e-3,
             nb::arg("measurement_variance") = 0.25,
             nb::arg("fillna") = true)
        .def("update", &KalmanExtremumTrend::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        .def("last", &KalmanExtremumTrend::last)
        RTTA_ADVANCE3(KalmanExtremumTrend, close, high, low)
        RTTA_REPLAY3(KalmanExtremumTrend, close, high, low)
        RTTA_FIELD3(KalmanExtremumTrend, trend, close, high, low)
        RTTA_FIELD3(KalmanExtremumTrend, oscillator, close, high, low)
        RTTA_FIELD3(KalmanExtremumTrend, signal, close, high, low)
        .def("replay_update_outputs", [](KalmanExtremumTrend &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("replay_update_outputs", [](KalmanExtremumTrend &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](KalmanExtremumTrend &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](KalmanExtremumTrend &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](KalmanExtremumTrend &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return indicator.batch_array(close, high, low);
                });
            }
            std::vector<double> trend = make_record_output(records);
            std::vector<double> oscillator;
            std::vector<double> signal;
            oscillator.reserve(trend.capacity());
            signal.reserve(trend.capacity());
            for (nb::handle record : records) {
                const KalmanExtremumTrendResult out = self.update(
                    record_value(record, "close", 0),
                    record_value(record, "high", 1),
                    record_value(record, "low", 2));
                trend.push_back(out.trend);
                oscillator.push_back(out.oscillator);
                signal.push_back(out.signal);
            }
            return KalmanExtremumTrendBatchResult{
                make_array(std::move(trend)),
                make_array(std::move(oscillator)),
                make_array(std::move(signal)),
            };
        }, nb::arg("records"));

    nb::class_<AlphaBetaGammaTrackingFilter>(m, "AlphaBetaGammaTrackingFilter")
        .def(nb::init<double, double, double, double, double, double, double, bool>(),
             nb::arg("alpha") = 0.65,
             nb::arg("beta") = 0.25,
             nb::arg("gamma") = 0.05,
             nb::arg("dt") = 1.0,
             nb::arg("initial_price") = nan(),
             nb::arg("initial_velocity") = 0.0,
             nb::arg("initial_acceleration") = 0.0,
             nb::arg("fillna") = true)
        .def("update", &AlphaBetaGammaTrackingFilter::update, nb::arg("close"))
        .def("last", &AlphaBetaGammaTrackingFilter::last)
        RTTA_ADVANCE1(AlphaBetaGammaTrackingFilter, close)
        RTTA_REPLAY1(AlphaBetaGammaTrackingFilter, close)
        RTTA_FIELD1(AlphaBetaGammaTrackingFilter, price, close)
        RTTA_FIELD1(AlphaBetaGammaTrackingFilter, velocity, close)
        RTTA_FIELD1(AlphaBetaGammaTrackingFilter, acceleration, close)
        RTTA_FIELD1(AlphaBetaGammaTrackingFilter, residual, close)
        .def("replay_update_outputs", [](AlphaBetaGammaTrackingFilter &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](AlphaBetaGammaTrackingFilter &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](AlphaBetaGammaTrackingFilter &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](AlphaBetaGammaTrackingFilter &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](AlphaBetaGammaTrackingFilter &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> price = make_record_output(records);
            std::vector<double> velocity;
            std::vector<double> acceleration;
            std::vector<double> residual;
            velocity.reserve(price.capacity());
            acceleration.reserve(price.capacity());
            residual.reserve(price.capacity());
            for (nb::handle record : records) {
                const AlphaBetaGammaTrackingFilterResult out = self.update(scalar_or_record_value(record, "close", 0));
                price.push_back(out.price);
                velocity.push_back(out.velocity);
                acceleration.push_back(out.acceleration);
                residual.push_back(out.residual);
            }
            return AlphaBetaGammaTrackingFilterBatchResult{
                make_array(std::move(price)),
                make_array(std::move(velocity)),
                make_array(std::move(acceleration)),
                make_array(std::move(residual)),
            };
        }, nb::arg("records"));

    nb::class_<InteractingMultipleModelFilter>(m, "InteractingMultipleModelFilter")
        .def(nb::init<double, double, double, double, double, double, double, double, double, bool>(),
             nb::arg("initial_price") = nan(),
             nb::arg("initial_velocity") = 0.0,
             nb::arg("dt") = 1.0,
             nb::arg("low_vol_process_variance") = 1.0e-5,
             nb::arg("high_vol_process_variance") = 1.0e-2,
             nb::arg("trend_process_variance") = 1.0e-4,
             nb::arg("chop_process_variance") = 1.0e-3,
             nb::arg("measurement_variance") = 0.25,
             nb::arg("stickiness") = 0.96,
             nb::arg("fillna") = true)
        .def("update", &InteractingMultipleModelFilter::update, nb::arg("close"))
        .def("last", &InteractingMultipleModelFilter::last)
        RTTA_ADVANCE1(InteractingMultipleModelFilter, close)
        RTTA_REPLAY1(InteractingMultipleModelFilter, close)
        RTTA_FIELD1(InteractingMultipleModelFilter, value, close)
        RTTA_FIELD1(InteractingMultipleModelFilter, velocity, close)
        RTTA_FIELD1(InteractingMultipleModelFilter, low_vol_probability, close)
        RTTA_FIELD1(InteractingMultipleModelFilter, high_vol_probability, close)
        RTTA_FIELD1(InteractingMultipleModelFilter, trend_probability, close)
        RTTA_FIELD1(InteractingMultipleModelFilter, chop_probability, close)
        .def("replay_update_outputs", [](InteractingMultipleModelFilter &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](InteractingMultipleModelFilter &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](InteractingMultipleModelFilter &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](InteractingMultipleModelFilter &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](InteractingMultipleModelFilter &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> value = make_record_output(records);
            std::vector<double> velocity;
            std::vector<double> low_vol_probability;
            std::vector<double> high_vol_probability;
            std::vector<double> trend_probability;
            std::vector<double> chop_probability;
            velocity.reserve(value.capacity());
            low_vol_probability.reserve(value.capacity());
            high_vol_probability.reserve(value.capacity());
            trend_probability.reserve(value.capacity());
            chop_probability.reserve(value.capacity());
            for (nb::handle record : records) {
                const InteractingMultipleModelFilterResult out = self.update(scalar_or_record_value(record, "close", 0));
                value.push_back(out.value);
                velocity.push_back(out.velocity);
                low_vol_probability.push_back(out.low_vol_probability);
                high_vol_probability.push_back(out.high_vol_probability);
                trend_probability.push_back(out.trend_probability);
                chop_probability.push_back(out.chop_probability);
            }
            return InteractingMultipleModelFilterBatchResult{
                make_array(std::move(value)),
                make_array(std::move(velocity)),
                make_array(std::move(low_vol_probability)),
                make_array(std::move(high_vol_probability)),
                make_array(std::move(trend_probability)),
                make_array(std::move(chop_probability)),
            };
        }, nb::arg("records"));

    nb::class_<ParticleFilterTrend>(m, "ParticleFilterTrend")
        .def(nb::init<int, double, double, double, double, double, double, double, std::uint64_t, bool>(),
             nb::arg("particles") = 128,
             nb::arg("initial_price") = nan(),
             nb::arg("initial_velocity") = 0.0,
             nb::arg("dt") = 1.0,
             nb::arg("process_position_scale") = 0.05,
             nb::arg("process_velocity_scale") = 0.01,
             nb::arg("measurement_scale") = 0.5,
             nb::arg("resample_threshold") = 0.5,
             nb::arg("seed") = 1,
             nb::arg("fillna") = true)
        .def("update", &ParticleFilterTrend::update, nb::arg("close"))
        .def("last", &ParticleFilterTrend::last)
        RTTA_ADVANCE1(ParticleFilterTrend, close)
        RTTA_REPLAY1(ParticleFilterTrend, close)
        RTTA_FIELD1(ParticleFilterTrend, trend, close)
        RTTA_FIELD1(ParticleFilterTrend, velocity, close)
        RTTA_FIELD1(ParticleFilterTrend, signal, close)
        RTTA_FIELD1(ParticleFilterTrend, effective_sample_size, close)
        .def("replay_update_outputs", [](ParticleFilterTrend &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](ParticleFilterTrend &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](ParticleFilterTrend &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](ParticleFilterTrend &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](ParticleFilterTrend &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> trend = make_record_output(records);
            std::vector<double> velocity;
            std::vector<double> signal;
            std::vector<double> effective_sample_size;
            velocity.reserve(trend.capacity());
            signal.reserve(trend.capacity());
            effective_sample_size.reserve(trend.capacity());
            for (nb::handle record : records) {
                const ParticleFilterTrendResult out = self.update(scalar_or_record_value(record, "close", 0));
                trend.push_back(out.trend);
                velocity.push_back(out.velocity);
                signal.push_back(out.signal);
                effective_sample_size.push_back(out.effective_sample_size);
            }
            return ParticleFilterTrendBatchResult{
                make_array(std::move(trend)),
                make_array(std::move(velocity)),
                make_array(std::move(signal)),
                make_array(std::move(effective_sample_size)),
            };
        }, nb::arg("records"));

    nb::class_<SavitzkyGolayFilter>(m, "SavitzkyGolayFilter")
        .def(nb::init<int, int, double, bool>(),
             nb::arg("window") = 7,
             nb::arg("polynomial_order") = 2,
             nb::arg("dt") = 1.0,
             nb::arg("fillna") = true)
        .def("update", &SavitzkyGolayFilter::update, nb::arg("close"))
        .def("last", &SavitzkyGolayFilter::last)
        RTTA_ADVANCE1(SavitzkyGolayFilter, close)
        RTTA_REPLAY1(SavitzkyGolayFilter, close)
        RTTA_FIELD1(SavitzkyGolayFilter, smooth, close)
        RTTA_FIELD1(SavitzkyGolayFilter, first_derivative, close)
        RTTA_FIELD1(SavitzkyGolayFilter, second_derivative, close)
        .def("replay_update_outputs", [](SavitzkyGolayFilter &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](SavitzkyGolayFilter &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](SavitzkyGolayFilter &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](SavitzkyGolayFilter &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](SavitzkyGolayFilter &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> smooth = make_record_output(records);
            std::vector<double> first_derivative;
            std::vector<double> second_derivative;
            first_derivative.reserve(smooth.capacity());
            second_derivative.reserve(smooth.capacity());
            for (nb::handle record : records) {
                const SavitzkyGolayFilterResult out = self.update(scalar_or_record_value(record, "close", 0));
                smooth.push_back(out.smooth);
                first_derivative.push_back(out.first_derivative);
                second_derivative.push_back(out.second_derivative);
            }
            return SavitzkyGolayFilterBatchResult{
                make_array(std::move(smooth)),
                make_array(std::move(first_derivative)),
                make_array(std::move(second_derivative)),
            };
        }, nb::arg("records"));

    nb::class_<NadarayaWatsonEnvelope>(m, "NadarayaWatsonEnvelope")
        .def(nb::init<int, double, double, bool>(),
             nb::arg("window") = 64,
             nb::arg("bandwidth") = 8.0,
             nb::arg("multiplier") = 2.0,
             nb::arg("fillna") = true)
        .def("update", &NadarayaWatsonEnvelope::update, nb::arg("close"))
        .def("last", &NadarayaWatsonEnvelope::last)
        RTTA_ADVANCE1(NadarayaWatsonEnvelope, close)
        RTTA_REPLAY1(NadarayaWatsonEnvelope, close)
        RTTA_FIELD1(NadarayaWatsonEnvelope, middle, close)
        RTTA_FIELD1(NadarayaWatsonEnvelope, upper, close)
        RTTA_FIELD1(NadarayaWatsonEnvelope, lower, close)
        .def("replay_update_outputs", [](NadarayaWatsonEnvelope &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](NadarayaWatsonEnvelope &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](NadarayaWatsonEnvelope &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](NadarayaWatsonEnvelope &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](NadarayaWatsonEnvelope &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> middle = make_record_output(records);
            std::vector<double> upper;
            std::vector<double> lower;
            upper.reserve(middle.capacity());
            lower.reserve(middle.capacity());
            for (nb::handle record : records) {
                const NadarayaWatsonEnvelopeResult out = self.update(scalar_or_record_value(record, "close", 0));
                middle.push_back(out.middle);
                upper.push_back(out.upper);
                lower.push_back(out.lower);
            }
            return NadarayaWatsonEnvelopeBatchResult{
                make_array(std::move(middle)),
                make_array(std::move(upper)),
                make_array(std::move(lower)),
            };
        }, nb::arg("records"));

    nb::class_<GaussianProcessRegressionBands>(m, "GaussianProcessRegressionBands")
        .def(nb::init<int, double, double, double, double, bool>(),
             nb::arg("window") = 32,
             nb::arg("length_scale") = 8.0,
             nb::arg("signal_variance") = 1.0,
             nb::arg("noise_variance") = 0.25,
             nb::arg("multiplier") = 2.0,
             nb::arg("fillna") = true)
        .def("update", &GaussianProcessRegressionBands::update, nb::arg("close"))
        .def("last", &GaussianProcessRegressionBands::last)
        RTTA_ADVANCE1(GaussianProcessRegressionBands, close)
        RTTA_REPLAY1(GaussianProcessRegressionBands, close)
        RTTA_FIELD1(GaussianProcessRegressionBands, middle, close)
        RTTA_FIELD1(GaussianProcessRegressionBands, upper, close)
        RTTA_FIELD1(GaussianProcessRegressionBands, lower, close)
        .def("replay_update_outputs", [](GaussianProcessRegressionBands &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](GaussianProcessRegressionBands &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](GaussianProcessRegressionBands &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](GaussianProcessRegressionBands &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](GaussianProcessRegressionBands &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> middle = make_record_output(records);
            std::vector<double> upper;
            std::vector<double> lower;
            upper.reserve(middle.capacity());
            lower.reserve(middle.capacity());
            for (nb::handle record : records) {
                const GaussianProcessRegressionBandsResult out = self.update(scalar_or_record_value(record, "close", 0));
                middle.push_back(out.middle);
                upper.push_back(out.upper);
                lower.push_back(out.lower);
            }
            return GaussianProcessRegressionBandsBatchResult{
                make_array(std::move(middle)),
                make_array(std::move(upper)),
                make_array(std::move(lower)),
            };
        }, nb::arg("records"));

    nb::class_<ZigZagSwingDetector>(m, "ZigZagSwingDetector")
        .def(nb::init<double>(), nb::arg("percent_change") = 5.0)
        .def("update", &ZigZagSwingDetector::update, nb::arg("close"))
        .def("last", &ZigZagSwingDetector::last)
        RTTA_ADVANCE1(ZigZagSwingDetector, close)
        RTTA_REPLAY1(ZigZagSwingDetector, close)
        RTTA_FIELD1(ZigZagSwingDetector, value, close)
        RTTA_FIELD1(ZigZagSwingDetector, direction, close)
        RTTA_FIELD1(ZigZagSwingDetector, pivot, close)
        RTTA_FIELD1(ZigZagSwingDetector, pivot_index, close)
        .def("replay_update_outputs", [](ZigZagSwingDetector &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](ZigZagSwingDetector &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](ZigZagSwingDetector &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](ZigZagSwingDetector &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](ZigZagSwingDetector &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> value = make_record_output(records);
            std::vector<double> direction;
            std::vector<double> pivot;
            std::vector<double> pivot_index;
            direction.reserve(value.capacity());
            pivot.reserve(value.capacity());
            pivot_index.reserve(value.capacity());
            for (nb::handle record : records) {
                const ZigZagSwingDetectorResult out = self.update(scalar_or_record_value(record, "close", 0));
                value.push_back(out.value);
                direction.push_back(out.direction);
                pivot.push_back(out.pivot);
                pivot_index.push_back(out.pivot_index);
            }
            return ZigZagSwingDetectorBatchResult{
                make_array(std::move(value)),
                make_array(std::move(direction)),
                make_array(std::move(pivot)),
                make_array(std::move(pivot_index)),
            };
        }, nb::arg("records"));

    nb::class_<RenkoBrickGenerator>(m, "RenkoBrickGenerator")
        .def(nb::init<double>(), nb::arg("brick_size") = 1.0)
        .def("update", &RenkoBrickGenerator::update, nb::arg("close"))
        .def("last", &RenkoBrickGenerator::last)
        RTTA_ADVANCE1(RenkoBrickGenerator, close)
        RTTA_REPLAY1(RenkoBrickGenerator, close)
        RTTA_FIELD1(RenkoBrickGenerator, brick_open, close)
        RTTA_FIELD1(RenkoBrickGenerator, brick_close, close)
        RTTA_FIELD1(RenkoBrickGenerator, direction, close)
        RTTA_FIELD1(RenkoBrickGenerator, bricks, close)
        RTTA_FIELD1(RenkoBrickGenerator, reversal, close)
        .def("replay_update_outputs", [](RenkoBrickGenerator &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("replay_update_outputs", [](RenkoBrickGenerator &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](RenkoBrickGenerator &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](RenkoBrickGenerator &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](RenkoBrickGenerator &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            std::vector<double> brick_open = make_record_output(records);
            std::vector<double> brick_close;
            std::vector<double> direction;
            std::vector<double> bricks;
            std::vector<double> reversal;
            brick_close.reserve(brick_open.capacity());
            direction.reserve(brick_open.capacity());
            bricks.reserve(brick_open.capacity());
            reversal.reserve(brick_open.capacity());
            for (nb::handle record : records) {
                const RenkoBrickGeneratorResult out = self.update(scalar_or_record_value(record, "close", 0));
                brick_open.push_back(out.brick_open);
                brick_close.push_back(out.brick_close);
                direction.push_back(out.direction);
                bricks.push_back(out.bricks);
                reversal.push_back(out.reversal);
            }
            return RenkoBrickGeneratorBatchResult{
                make_array(std::move(brick_open)),
                make_array(std::move(brick_close)),
                make_array(std::move(direction)),
                make_array(std::move(bricks)),
                make_array(std::move(reversal)),
            };
        }, nb::arg("records"));

    nb::class_<HeikinAshiTransform>(m, "HeikinAshiTransform")
        .def(nb::init<>())
        .def("update", &HeikinAshiTransform::update, nb::arg("open"), nb::arg("high"), nb::arg("low"), nb::arg("close"))
        .def("last", &HeikinAshiTransform::last)
        RTTA_ADVANCE4(HeikinAshiTransform, open, high, low, close)
        RTTA_REPLAY4(HeikinAshiTransform, open, high, low, close)
        RTTA_FIELD4(HeikinAshiTransform, open, open, high, low, close)
        RTTA_FIELD4(HeikinAshiTransform, high, open, high, low, close)
        RTTA_FIELD4(HeikinAshiTransform, low, open, high, low, close)
        RTTA_FIELD4(HeikinAshiTransform, close, open, high, low, close)
        .def("replay_update_outputs", [](HeikinAshiTransform &self, const InputArray &open, const InputArray &high, const InputArray &low, const InputArray &close) {
            return self.batch_array(open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("replay_update_outputs", [](HeikinAshiTransform &self, const FloatInputArray &open, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &close) {
            return self.batch_array(open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("batch", [](HeikinAshiTransform &self, const InputArray &open, const InputArray &high, const InputArray &low, const InputArray &close) {
            return self.batch_array(open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("batch", [](HeikinAshiTransform &self, const FloatInputArray &open, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &close) {
            return self.batch_array(open, high, low, close);
        }, array_arg("open"), array_arg("high"), array_arg("low"), array_arg("close"))
        .def("batch", [](HeikinAshiTransform &self, nb::iterable records) {
            if (table_has_column(records, "open")) {
                return dispatch_table4(self, records, "open", "high", "low", "close", [](auto &indicator, const auto &open, const auto &high, const auto &low, const auto &close) {
                    return indicator.batch_array(open, high, low, close);
                });
            }
            std::vector<double> out_open = make_record_output(records);
            std::vector<double> out_high;
            std::vector<double> out_low;
            std::vector<double> out_close;
            out_high.reserve(out_open.capacity());
            out_low.reserve(out_open.capacity());
            out_close.reserve(out_open.capacity());
            for (nb::handle record : records) {
                const HeikinAshiTransformResult out = self.update(
                    record_value(record, "open", 0),
                    record_value(record, "high", 1),
                    record_value(record, "low", 2),
                    record_value(record, "close", 3));
                out_open.push_back(out.open);
                out_high.push_back(out.high);
                out_low.push_back(out.low);
                out_close.push_back(out.close);
            }
            return HeikinAshiTransformBatchResult{
                make_array(std::move(out_open)),
                make_array(std::move(out_high)),
                make_array(std::move(out_low)),
                make_array(std::move(out_close)),
            };
        }, nb::arg("records"));

    nb::class_<MesaAdaptiveMovingAverage>(m, "MesaAdaptiveMovingAverage")
        .def(nb::init<double, double, bool>(),
             nb::arg("fast_limit") = 0.5,
             nb::arg("slow_limit") = 0.05,
             nb::arg("fillna") = true)
        .def("update", &MesaAdaptiveMovingAverage::update, nb::arg("value"))
        .def("last", &MesaAdaptiveMovingAverage::last)
        RTTA_ADVANCE1(MesaAdaptiveMovingAverage, value)
        RTTA_REPLAY1(MesaAdaptiveMovingAverage, value)
        RTTA_FIELD1(MesaAdaptiveMovingAverage, mama, value)
        RTTA_FIELD1(MesaAdaptiveMovingAverage, fama, value)
        .def("replay_update_outputs", [](MesaAdaptiveMovingAverage &self, const InputArray &value) {
            return self.batch_array(value);
        }, array_arg("value"))
        .def("replay_update_outputs", [](MesaAdaptiveMovingAverage &self, const FloatInputArray &value) {
            return self.batch_array(value);
        }, array_arg("value"))
        .def("batch", [](MesaAdaptiveMovingAverage &self, const InputArray &value) {
            return self.batch_array(value);
        }, array_arg("value"))
        .def("batch", [](MesaAdaptiveMovingAverage &self, const FloatInputArray &value) {
            return self.batch_array(value);
        }, array_arg("value"))
        .def("batch", [](MesaAdaptiveMovingAverage &self, nb::iterable records) {
            if (table_has_column(records, "value")) {
                return dispatch_table1(self, records, "value", [](auto &indicator, const auto &value) {
                    return indicator.batch_array(value);
                });
            }
            std::vector<double> mama = make_record_output(records);
            std::vector<double> fama;
            fama.reserve(mama.capacity());
            for (nb::handle record : records) {
                const MesaAdaptiveMovingAverageResult out = self.update(scalar_or_record_value(record, "value", 0));
                mama.push_back(out.mama);
                fama.push_back(out.fama);
            }
            return MesaAdaptiveMovingAverageBatchResult{make_array(std::move(mama)), make_array(std::move(fama))};
        }, nb::arg("records"));

    nb::class_<EhlersOptimalTrackingFilter>(m, "EhlersOptimalTrackingFilter")
        .def(nb::init<bool>(), nb::arg("fillna") = true)
        .def("update", &EhlersOptimalTrackingFilter::update, nb::arg("high"), nb::arg("low"))
        .def("last", &EhlersOptimalTrackingFilter::last)
        RTTA_ADVANCE2(EhlersOptimalTrackingFilter, high, low)
        RTTA_REPLAY2(EhlersOptimalTrackingFilter, high, low)
        RTTA_BATCH2_ARRAY(EhlersOptimalTrackingFilter, high, low, "high", "low");

    nb::class_<VariableIndexDynamicAverage>(m, "VariableIndexDynamicAverage")
        .def(nb::init<int, int, bool>(),
             nb::arg("cmo_window") = 9,
             nb::arg("ema_window") = 12,
             nb::arg("fillna") = true)
        .def("update", &VariableIndexDynamicAverage::update, nb::arg("close"))
        RTTA_ADVANCE1(VariableIndexDynamicAverage, close)
        RTTA_REPLAY1(VariableIndexDynamicAverage, close)
        .def("batch", [](VariableIndexDynamicAverage &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](VariableIndexDynamicAverage &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](VariableIndexDynamicAverage &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            return batch_records_one(self, records, "close");
        }, nb::arg("records"));

    nb::class_<KeltnerChannel>(m, "KeltnerChannel")
        .def(nb::init<double, double, bool, double>(),
             nb::arg("span") = 20.0,
             nb::arg("window_atr") = 20.0,
             nb::arg("fillna") = false,
             nb::arg("multiplier") = 2.0)
        .def("update", &KeltnerChannel::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(KeltnerChannel, close, high, low)
        RTTA_REPLAY3(KeltnerChannel, close, high, low)
        RTTA_FIELD3(KeltnerChannel, middle, close, high, low)
        RTTA_FIELD3(KeltnerChannel, upper, close, high, low)
        RTTA_FIELD3(KeltnerChannel, lower, close, high, low)
        RTTA_REPLAY_OUTPUTS3(KeltnerChannel, close, high, low, batch_keltner)
        .def("batch", [](KeltnerChannel &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return batch_keltner(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](KeltnerChannel &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return batch_keltner(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](KeltnerChannel &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return batch_keltner(indicator, close, high, low);
                });
            }

            std::vector<double> middle = make_record_output(records);
            std::vector<double> upper;
            std::vector<double> lower;
            upper.reserve(middle.capacity());
            lower.reserve(middle.capacity());
            for (nb::handle record : records) {
                const KeltnerChannelResult out = self.update(
                    record_value(record, "close", 0),
                    record_value(record, "high", 1),
                    record_value(record, "low", 2));
                middle.push_back(out.middle);
                upper.push_back(out.upper);
                lower.push_back(out.lower);
            }
            return KeltnerChannelBatchResult{
                make_array(std::move(middle)),
                make_array(std::move(upper)),
                make_array(std::move(lower)),
            };
        }, nb::arg("records"));

    nb::class_<KeltnerChannelOriginal>(m, "KeltnerChannelOriginal")
        .def(nb::init<int, bool>(), nb::arg("window") = 20, nb::arg("fillna") = false)
        .def("update", &KeltnerChannelOriginal::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(KeltnerChannelOriginal, close, high, low)
        RTTA_REPLAY3(KeltnerChannelOriginal, close, high, low)
        RTTA_FIELD3(KeltnerChannelOriginal, middle, close, high, low)
        RTTA_FIELD3(KeltnerChannelOriginal, upper, close, high, low)
        RTTA_FIELD3(KeltnerChannelOriginal, lower, close, high, low)
        RTTA_REPLAY_OUTPUTS3(KeltnerChannelOriginal, close, high, low, batch_keltner)
        .def("batch", [](KeltnerChannelOriginal &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return batch_keltner(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](KeltnerChannelOriginal &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return batch_keltner(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](KeltnerChannelOriginal &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return batch_keltner(indicator, close, high, low);
                });
            }

            std::vector<double> middle = make_record_output(records);
            std::vector<double> upper;
            std::vector<double> lower;
            upper.reserve(middle.capacity());
            lower.reserve(middle.capacity());
            for (nb::handle record : records) {
                const KeltnerChannelResult out = self.update(
                    record_value(record, "close", 0),
                    record_value(record, "high", 1),
                    record_value(record, "low", 2));
                middle.push_back(out.middle);
                upper.push_back(out.upper);
                lower.push_back(out.lower);
            }
            return KeltnerChannelBatchResult{
                make_array(std::move(middle)),
                make_array(std::move(upper)),
                make_array(std::move(lower)),
            };
        }, nb::arg("records"));

    nb::class_<KSTOscillator>(m, "KSTOscillator")
        .def(nb::init<int, int, int, int, int, int, int, int, int, bool>(),
             nb::arg("roc1") = 10,
             nb::arg("roc2") = 15,
             nb::arg("roc3") = 20,
             nb::arg("roc4") = 30,
             nb::arg("window1") = 10,
             nb::arg("window2") = 10,
             nb::arg("window3") = 10,
             nb::arg("window4") = 15,
             nb::arg("signal") = 9,
             nb::arg("fillna") = true)
        .def("update", &KSTOscillator::update, nb::arg("close"))
        RTTA_ADVANCE1(KSTOscillator, close)
        RTTA_REPLAY1(KSTOscillator, close)
        RTTA_FIELD1(KSTOscillator, kst, close)
        RTTA_FIELD1(KSTOscillator, signal, close)
        RTTA_FIELD1(KSTOscillator, difference, close)
        RTTA_REPLAY_OUTPUTS1(KSTOscillator, close, batch_kst)
        .def("batch", &KSTOscillator::batch, array_arg("close"))
        .def("batch", [](KSTOscillator &self, const FloatInputArray &close) {
            return batch_kst(self, close);
        }, array_arg("close"))
        .def("batch", [](KSTOscillator &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return batch_kst(indicator, close);
                });
            }

            std::vector<double> kst = make_record_output(records);
            std::vector<double> signal;
            std::vector<double> difference;
            signal.reserve(kst.capacity());
            difference.reserve(kst.capacity());
            for (nb::handle record : records) {
                const KSTOscillatorResult out = self.update(record_value(record, "close", 0));
                kst.push_back(out.kst);
                signal.push_back(out.signal);
                difference.push_back(out.difference);
            }
            return KSTOscillatorBatchResult{
                make_array(std::move(kst)),
                make_array(std::move(signal)),
                make_array(std::move(difference)),
            };
        }, nb::arg("records"));

    nb::class_<LinearRegression>(m, "LinearRegression")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &LinearRegression::update, nb::arg("value"))
        RTTA_ADVANCE1(LinearRegression, value)
        RTTA_REPLAY1(LinearRegression, value)
        RTTA_BATCH1(LinearRegression, value, "value");

    nb::class_<LinearRegressionAngle>(m, "LinearRegressionAngle")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &LinearRegressionAngle::update, nb::arg("value"))
        RTTA_ADVANCE1(LinearRegressionAngle, value)
        RTTA_REPLAY1(LinearRegressionAngle, value)
        RTTA_BATCH1(LinearRegressionAngle, value, "value");

    nb::class_<LinearRegressionIntercept>(m, "LinearRegressionIntercept")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &LinearRegressionIntercept::update, nb::arg("value"))
        RTTA_ADVANCE1(LinearRegressionIntercept, value)
        RTTA_REPLAY1(LinearRegressionIntercept, value)
        RTTA_BATCH1(LinearRegressionIntercept, value, "value");

    nb::class_<LinearRegressionSlope>(m, "LinearRegressionSlope")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &LinearRegressionSlope::update, nb::arg("value"))
        RTTA_ADVANCE1(LinearRegressionSlope, value)
        RTTA_REPLAY1(LinearRegressionSlope, value)
        RTTA_BATCH1(LinearRegressionSlope, value, "value");

    nb::class_<MACD>(m, "MACD")
        .def(nb::init<int, int, int, bool>(), nb::arg("a") = 12, nb::arg("b") = 26, nb::arg("c") = 9, nb::arg("fillna") = false)
        .def("update", &MACD::update, nb::arg("value"))
        RTTA_ADVANCE1(MACD, value)
        RTTA_REPLAY1(MACD, value)
        .def("batch", &MACD::batch, array_arg("input"))
        .def("batch", [](MACD &self, const FloatInputArray &input) {
            return batch_update1(self, input);
        }, array_arg("input"))
        .def("batch", [](MACD &self, nb::iterable records) {
            return batch_records_one(self, records, "input");
        }, nb::arg("records"));

    nb::class_<MACDFix>(m, "MACDFix")
        .def(nb::init<int, bool>(), nb::arg("signal") = 9, nb::arg("fillna") = false)
        .def("update", &MACDFix::update, nb::arg("close"))
        RTTA_ADVANCE1(MACDFix, close)
        RTTA_REPLAY1(MACDFix, close)
        RTTA_BATCH1(MACDFix, close, "close");

    nb::class_<MassIndex>(m, "MassIndex")
        .def(nb::init<int, int, int, bool>(),
             nb::arg("single") = 9,
             nb::arg("double") = 9,
             nb::arg("summation") = 25,
             nb::arg("fillna") = false)
        .def("update", &MassIndex::update, nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE2(MassIndex, high, low)
        RTTA_REPLAY2(MassIndex, high, low)
        RTTA_BATCH2(MassIndex, high, low, "high", "low");

    nb::class_<HighIndex>(m, "HighIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 30, nb::arg("fillna") = true)
        .def("update", &HighIndex::update, nb::arg("value"))
        RTTA_ADVANCE1(HighIndex, value)
        RTTA_REPLAY1(HighIndex, value)
        RTTA_BATCH1_ARRAY(HighIndex, value, "value");

    nb::class_<MedianPrice>(m, "MedianPrice")
        .def(nb::init<>())
        .def("update", &MedianPrice::update, nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE2(MedianPrice, high, low)
        RTTA_REPLAY2(MedianPrice, high, low)
        RTTA_BATCH2(MedianPrice, high, low, "high", "low");

    nb::class_<MoneyFlowIndex>(m, "MoneyFlowIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &MoneyFlowIndex::update, nb::arg("close"), nb::arg("high"), nb::arg("low"), nb::arg("volume"))
        RTTA_ADVANCE4(MoneyFlowIndex, close, high, low, volume)
        RTTA_REPLAY4(MoneyFlowIndex, close, high, low, volume)
        .def("batch", [](MoneyFlowIndex &self, const InputArray &close, const InputArray &high, const InputArray &low, const InputArray &volume) {
            return self.batch_array(close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](MoneyFlowIndex &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &volume) {
            return self.batch_array(close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](MoneyFlowIndex &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table4(self, records, "close", "high", "low", "volume", [](auto &indicator, const auto &close, const auto &high, const auto &low, const auto &volume) {
                    return indicator.batch_array(close, high, low, volume);
                });
            }
            return batch_records_four(self, records, "close", "high", "low", "volume");
        }, nb::arg("records"));

    nb::class_<MidPoint>(m, "MidPoint")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &MidPoint::update, nb::arg("value"))
        RTTA_ADVANCE1(MidPoint, value)
        RTTA_REPLAY1(MidPoint, value)
        RTTA_BATCH1_ARRAY(MidPoint, value, "value");

    nb::class_<MidPrice>(m, "MidPrice")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &MidPrice::update, nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE2(MidPrice, high, low)
        RTTA_REPLAY2(MidPrice, high, low)
        RTTA_BATCH2_ARRAY(MidPrice, high, low, "high", "low");

    nb::class_<LowIndex>(m, "LowIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 30, nb::arg("fillna") = true)
        .def("update", &LowIndex::update, nb::arg("value"))
        RTTA_ADVANCE1(LowIndex, value)
        RTTA_REPLAY1(LowIndex, value)
        RTTA_BATCH1_ARRAY(LowIndex, value, "value");

    nb::class_<HighLow>(m, "HighLow")
        .def(nb::init<int, bool>(), nb::arg("window") = 30, nb::arg("fillna") = true)
        .def("update", &HighLow::update, nb::arg("value"))
        RTTA_ADVANCE1(HighLow, value)
        RTTA_REPLAY1(HighLow, value)
        RTTA_FIELD1(HighLow, min, value)
        RTTA_FIELD1(HighLow, max, value)
        RTTA_REPLAY_OUTPUTS1(HighLow, value, batch_high_low)
        .def("batch", [](HighLow &self, const InputArray &value) {
            return batch_high_low(self, value);
        }, array_arg("value"))
        .def("batch", [](HighLow &self, const FloatInputArray &value) {
            return batch_high_low(self, value);
        }, array_arg("value"))
        .def("batch", [](HighLow &self, nb::iterable records) {
            if (table_has_column(records, "value")) {
                return dispatch_table1(self, records, "value", [](auto &indicator, const auto &value) {
                    return batch_high_low(indicator, value);
                });
            }

            std::vector<double> min_values = make_record_output(records);
            std::vector<double> max_values;
            max_values.reserve(min_values.capacity());
            for (nb::handle record : records) {
                const HighLowResult out = self.update(record_value(record, "value", 0));
                min_values.push_back(out.min);
                max_values.push_back(out.max);
            }
            return HighLowBatchResult{make_array(std::move(min_values)), make_array(std::move(max_values))};
        }, nb::arg("records"));

    nb::class_<HighLowIndex>(m, "HighLowIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 30, nb::arg("fillna") = true)
        .def("update", &HighLowIndex::update, nb::arg("value"))
        RTTA_ADVANCE1(HighLowIndex, value)
        RTTA_REPLAY1(HighLowIndex, value)
        RTTA_FIELD1(HighLowIndex, min_index, value)
        RTTA_FIELD1(HighLowIndex, max_index, value)
        RTTA_REPLAY_OUTPUTS1(HighLowIndex, value, batch_high_low_index)
        .def("batch", [](HighLowIndex &self, const InputArray &value) {
            return batch_high_low_index(self, value);
        }, array_arg("value"))
        .def("batch", [](HighLowIndex &self, const FloatInputArray &value) {
            return batch_high_low_index(self, value);
        }, array_arg("value"))
        .def("batch", [](HighLowIndex &self, nb::iterable records) {
            if (table_has_column(records, "value")) {
                return dispatch_table1(self, records, "value", [](auto &indicator, const auto &value) {
                    return batch_high_low_index(indicator, value);
                });
            }

            std::vector<double> min_index = make_record_output(records);
            std::vector<double> max_index;
            max_index.reserve(min_index.capacity());
            for (nb::handle record : records) {
                const HighLowIndexResult out = self.update(record_value(record, "value", 0));
                min_index.push_back(out.min_index);
                max_index.push_back(out.max_index);
            }
            return HighLowIndexBatchResult{make_array(std::move(min_index)), make_array(std::move(max_index))};
        }, nb::arg("records"));

    nb::class_<HullMovingAverage>(m, "HullMovingAverage")
        .def(nb::init<int, bool>(), nb::arg("window") = 30, nb::arg("fillna") = true)
        .def("update", &HullMovingAverage::update, nb::arg("value"))
        RTTA_ADVANCE1(HullMovingAverage, value)
        RTTA_REPLAY1(HullMovingAverage, value)
        RTTA_BATCH1_ARRAY(HullMovingAverage, value, "value");

    nb::class_<MinusDirectionalIndicator>(m, "MinusDirectionalIndicator")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &MinusDirectionalIndicator::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(MinusDirectionalIndicator, close, high, low)
        RTTA_REPLAY3(MinusDirectionalIndicator, close, high, low)
        RTTA_BATCH3_ARRAY(MinusDirectionalIndicator, close, high, low, "close", "high", "low");

    nb::class_<MinusDirectionalMovement>(m, "MinusDirectionalMovement")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &MinusDirectionalMovement::update, nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE2(MinusDirectionalMovement, high, low)
        RTTA_REPLAY2(MinusDirectionalMovement, high, low)
        RTTA_BATCH2_ARRAY(MinusDirectionalMovement, high, low, "high", "low");

    nb::class_<Momentum>(m, "Momentum")
        .def(nb::init<int, bool>(), nb::arg("window") = 10, nb::arg("fillna") = true)
        .def("update", &Momentum::update, nb::arg("close"))
        RTTA_ADVANCE1(Momentum, close)
        RTTA_REPLAY1(Momentum, close)
        .def("batch", [](Momentum &self, const InputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](Momentum &self, const FloatInputArray &close) {
            return self.batch_array(close);
        }, array_arg("close"))
        .def("batch", [](Momentum &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table1(self, records, "close", [](auto &indicator, const auto &close) {
                    return indicator.batch_array(close);
                });
            }
            return batch_records_one(self, records, "close");
        }, nb::arg("records"));

    nb::class_<NormalizedATR>(m, "NormalizedATR")
        .def(nb::init<double, bool>(), nb::arg("window") = 14.0, nb::arg("fillna") = true)
        .def("update", &NormalizedATR::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(NormalizedATR, close, high, low)
        RTTA_REPLAY3(NormalizedATR, close, high, low)
        RTTA_BATCH3_ARRAY(NormalizedATR, close, high, low, "close", "high", "low");

    nb::class_<OnBalanceVolume>(m, "OnBalanceVolume")
        .def(nb::init<>())
        .def("update", &OnBalanceVolume::update, nb::arg("close"), nb::arg("volume"))
        RTTA_ADVANCE2(OnBalanceVolume, close, volume)
        RTTA_REPLAY2(OnBalanceVolume, close, volume)
        RTTA_BATCH2(OnBalanceVolume, close, volume, "close", "volume");

    nb::class_<OrderFlowImbalance>(m, "OrderFlowImbalance")
        .def(nb::init<int, bool>(), nb::arg("window") = 1, nb::arg("fillna") = true)
        .def("update", &OrderFlowImbalance::update, nb::arg("bid_price"), nb::arg("bid_size"), nb::arg("ask_price"), nb::arg("ask_size"))
        .def("last", &OrderFlowImbalance::last)
        RTTA_ADVANCE4(OrderFlowImbalance, bid_price, bid_size, ask_price, ask_size)
        RTTA_REPLAY4(OrderFlowImbalance, bid_price, bid_size, ask_price, ask_size)
        .def("batch", [](OrderFlowImbalance &self, const InputArray &bid_price, const InputArray &bid_size, const InputArray &ask_price, const InputArray &ask_size) {
            return self.batch_array(bid_price, bid_size, ask_price, ask_size);
        }, array_arg("bid_price"), array_arg("bid_size"), array_arg("ask_price"), array_arg("ask_size"))
        .def("batch", [](OrderFlowImbalance &self, const FloatInputArray &bid_price, const FloatInputArray &bid_size, const FloatInputArray &ask_price, const FloatInputArray &ask_size) {
            return self.batch_array(bid_price, bid_size, ask_price, ask_size);
        }, array_arg("bid_price"), array_arg("bid_size"), array_arg("ask_price"), array_arg("ask_size"))
        .def("batch", [](OrderFlowImbalance &self, nb::iterable records) {
            if (table_has_column(records, "bid_price")) {
                return dispatch_table4(self, records, "bid_price", "bid_size", "ask_price", "ask_size", [](auto &indicator, const auto &bid_price, const auto &bid_size, const auto &ask_price, const auto &ask_size) {
                    return indicator.batch_array(bid_price, bid_size, ask_price, ask_size);
                });
            }
            return batch_records_four(self, records, "bid_price", "bid_size", "ask_price", "ask_size");
        }, nb::arg("records"));

    nb::class_<VPIN>(m, "VPIN")
        .def(nb::init<double, int, int, bool>(),
             nb::arg("bucket_volume") = 1000000.0,
             nb::arg("buckets") = 50,
             nb::arg("price_change_window") = 50,
             nb::arg("fillna") = true)
        .def("update", &VPIN::update, nb::arg("close"), nb::arg("volume"))
        .def("last", &VPIN::last)
        RTTA_ADVANCE2(VPIN, close, volume)
        RTTA_REPLAY2(VPIN, close, volume)
        RTTA_BATCH2_ARRAY(VPIN, close, volume, "close", "volume");

    nb::class_<KyleLambda>(m, "KyleLambda")
        .def(nb::init<int, double, bool>(), nb::arg("window") = 30, nb::arg("scale") = 1000000.0, nb::arg("fillna") = true)
        .def("update", &KyleLambda::update, nb::arg("close"), nb::arg("signed_dollar_volume"))
        .def("last", &KyleLambda::last)
        RTTA_ADVANCE2(KyleLambda, close, signed_dollar_volume)
        RTTA_REPLAY2(KyleLambda, close, signed_dollar_volume)
        RTTA_BATCH2_ARRAY(KyleLambda, close, signed_dollar_volume, "close", "signed_dollar_volume");

    nb::class_<AmihudIlliquidity>(m, "AmihudIlliquidity")
        .def(nb::init<int, double, bool>(), nb::arg("window") = 30, nb::arg("scale") = 1000000.0, nb::arg("fillna") = true)
        .def("update", &AmihudIlliquidity::update, nb::arg("close"), nb::arg("volume"))
        .def("last", &AmihudIlliquidity::last)
        RTTA_ADVANCE2(AmihudIlliquidity, close, volume)
        RTTA_REPLAY2(AmihudIlliquidity, close, volume)
        RTTA_BATCH2_ARRAY(AmihudIlliquidity, close, volume, "close", "volume");

    nb::class_<SpreadFeatures>(m, "SpreadFeatures")
        .def(nb::init<int, bool>(), nb::arg("realized_horizon") = 5, nb::arg("fillna") = true)
        .def("update", &SpreadFeatures::update, nb::arg("trade_price"), nb::arg("bid_price"), nb::arg("ask_price"))
        .def("last", &SpreadFeatures::last)
        RTTA_ADVANCE3(SpreadFeatures, trade_price, bid_price, ask_price)
        RTTA_REPLAY3(SpreadFeatures, trade_price, bid_price, ask_price)
        RTTA_FIELD3(SpreadFeatures, quoted_spread, trade_price, bid_price, ask_price)
        RTTA_FIELD3(SpreadFeatures, effective_spread, trade_price, bid_price, ask_price)
        RTTA_FIELD3(SpreadFeatures, realized_spread, trade_price, bid_price, ask_price)
        .def("replay_update_outputs", [](SpreadFeatures &self, const InputArray &trade_price, const InputArray &bid_price, const InputArray &ask_price) {
            return self.batch_array(trade_price, bid_price, ask_price);
        }, array_arg("trade_price"), array_arg("bid_price"), array_arg("ask_price"))
        .def("replay_update_outputs", [](SpreadFeatures &self, const FloatInputArray &trade_price, const FloatInputArray &bid_price, const FloatInputArray &ask_price) {
            return self.batch_array(trade_price, bid_price, ask_price);
        }, array_arg("trade_price"), array_arg("bid_price"), array_arg("ask_price"))
        .def("batch", [](SpreadFeatures &self, const InputArray &trade_price, const InputArray &bid_price, const InputArray &ask_price) {
            return self.batch_array(trade_price, bid_price, ask_price);
        }, array_arg("trade_price"), array_arg("bid_price"), array_arg("ask_price"))
        .def("batch", [](SpreadFeatures &self, const FloatInputArray &trade_price, const FloatInputArray &bid_price, const FloatInputArray &ask_price) {
            return self.batch_array(trade_price, bid_price, ask_price);
        }, array_arg("trade_price"), array_arg("bid_price"), array_arg("ask_price"))
        .def("batch", [](SpreadFeatures &self, nb::iterable records) {
            if (table_has_column(records, "trade_price")) {
                return dispatch_table3(self, records, "trade_price", "bid_price", "ask_price", [](auto &indicator, const auto &trade_price, const auto &bid_price, const auto &ask_price) {
                    return indicator.batch_array(trade_price, bid_price, ask_price);
                });
            }

            std::vector<double> quoted_spread = make_record_output(records);
            std::vector<double> effective_spread;
            std::vector<double> realized_spread;
            effective_spread.reserve(quoted_spread.capacity());
            realized_spread.reserve(quoted_spread.capacity());
            for (nb::handle record : records) {
                const SpreadFeaturesResult out = self.update(
                    record_value(record, "trade_price", 0),
                    record_value(record, "bid_price", 1),
                    record_value(record, "ask_price", 2));
                quoted_spread.push_back(out.quoted_spread);
                effective_spread.push_back(out.effective_spread);
                realized_spread.push_back(out.realized_spread);
            }
            return SpreadFeaturesBatchResult{
                make_array(std::move(quoted_spread)),
                make_array(std::move(effective_spread)),
                make_array(std::move(realized_spread)),
            };
        }, nb::arg("records"));

    nb::class_<ChaikinMoneyFlow>(m, "ChaikinMoneyFlow")
        .def(nb::init<int, bool>(), nb::arg("window") = 20, nb::arg("fillna") = true)
        .def("update", &ChaikinMoneyFlow::update, nb::arg("close"), nb::arg("high"), nb::arg("low"), nb::arg("volume"))
        RTTA_ADVANCE4(ChaikinMoneyFlow, close, high, low, volume)
        RTTA_REPLAY4(ChaikinMoneyFlow, close, high, low, volume)
        .def("batch", &ChaikinMoneyFlow::batch, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](ChaikinMoneyFlow &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &volume) {
            return batch_update4(self, close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](ChaikinMoneyFlow &self, nb::iterable records) {
            return batch_records_four(self, records, "close", "high", "low", "volume");
        }, nb::arg("records"));

    nb::class_<SMA>(m, "SMA")
        .def(nb::init<int, bool>(), nb::arg("window"), nb::arg("fillna") = false)
        .def("length", &SMA::length)
        .def("mean", &SMA::mean)
        .def("update", &SMA::update, nb::arg("value"))
        RTTA_ADVANCE1(SMA, value)
        RTTA_REPLAY1(SMA, value)
        .def("batch", &SMA::batch, array_arg("input"))
        .def("batch", [](SMA &self, const FloatInputArray &input) {
            return batch_update1(self, input);
        }, array_arg("input"))
        .def("batch", [](SMA &self, nb::iterable records) {
            return batch_records_one(self, records, "input");
        }, nb::arg("records"));

    nb::class_<PercentagePrice>(m, "PercentagePrice")
        .def(nb::init<double, double, double, bool>(),
             nb::arg("window_1") = 12.0,
             nb::arg("window_2") = 26.0,
             nb::arg("window_3") = 9.0,
             nb::arg("fillna") = false)
        .def("update", &PercentagePrice::update, nb::arg("close"))
        RTTA_ADVANCE1(PercentagePrice, close)
        RTTA_REPLAY1(PercentagePrice, close)
        RTTA_FIELD1(PercentagePrice, ppo, close)
        RTTA_FIELD1(PercentagePrice, signal, close)
        RTTA_FIELD1(PercentagePrice, histogram, close)
        RTTA_REPLAY_OUTPUTS1(PercentagePrice, close, batch_percentage_price)
        .def("batch", &PercentagePrice::batch, array_arg("close"))
        .def("batch", [](PercentagePrice &self, const FloatInputArray &close) {
            const std::size_t size = close.shape(0);
            std::vector<double> ppo(size);
            std::vector<double> signal(size);
            std::vector<double> histogram(size);
            const float *values = close.data();
            for (std::size_t i = 0; i < size; ++i) {
                const PercentagePriceResult out = self.update(static_cast<double>(values[i]));
                ppo[i] = out.ppo;
                signal[i] = out.signal;
                histogram[i] = out.histogram;
            }
            return PercentagePriceBatchResult{
                make_array(std::move(ppo)),
                make_array(std::move(signal)),
                make_array(std::move(histogram)),
            };
        }, array_arg("close"))
        .def("batch", &PercentagePrice::batch_records, nb::arg("records"))
        .def("batch_ppo", &PercentagePrice::batch_ppo, array_arg("close"))
        .def("batch_ppo", [](PercentagePrice &self, const FloatInputArray &close) {
            const std::size_t size = close.shape(0);
            std::vector<double> output(size);
            const float *values = close.data();
            for (std::size_t i = 0; i < size; ++i) {
                output[i] = self.update(static_cast<double>(values[i])).ppo;
            }
            return make_array(std::move(output));
        }, array_arg("close"))
        .def("batch_ppo", &PercentagePrice::batch_ppo_records, nb::arg("records"));

    nb::class_<PercentageVolume>(m, "PercentageVolume")
        .def(nb::init<double, double, double, bool>(),
             nb::arg("window_1") = 12.0,
             nb::arg("window_2") = 26.0,
             nb::arg("signal") = 9.0,
             nb::arg("fillna") = true)
        .def("update", &PercentageVolume::update, nb::arg("volume"))
        RTTA_ADVANCE1(PercentageVolume, volume)
        RTTA_REPLAY1(PercentageVolume, volume)
        RTTA_FIELD1(PercentageVolume, pvo, volume)
        RTTA_FIELD1(PercentageVolume, signal, volume)
        RTTA_FIELD1(PercentageVolume, histogram, volume)
        RTTA_REPLAY_OUTPUTS1(PercentageVolume, volume, batch_percentage_volume)
        .def("batch", &PercentageVolume::batch, array_arg("volume"))
        .def("batch", [](PercentageVolume &self, const FloatInputArray &volume) {
            const std::size_t size = volume.shape(0);
            std::vector<double> pvo(size);
            std::vector<double> signal(size);
            std::vector<double> histogram(size);
            const float *values = volume.data();
            for (std::size_t i = 0; i < size; ++i) {
                const PercentageVolumeResult out = self.update(static_cast<double>(values[i]));
                pvo[i] = out.pvo;
                signal[i] = out.signal;
                histogram[i] = out.histogram;
            }
            return PercentageVolumeBatchResult{
                make_array(std::move(pvo)),
                make_array(std::move(signal)),
                make_array(std::move(histogram)),
            };
        }, array_arg("volume"))
        .def("batch", [](PercentageVolume &self, nb::iterable records) {
            if (table_has_column(records, "volume")) {
                nb::object volume = table_column_array(records, "volume");
                switch (array_dtype(volume)) {
                    case InputDType::Float32: {
                        const FloatInputArray input = nb::cast<FloatInputArray>(volume);
                        const std::size_t size = input.shape(0);
                        std::vector<double> pvo(size);
                        std::vector<double> signal(size);
                        std::vector<double> histogram(size);
                        const float *values = input.data();
                        for (std::size_t i = 0; i < size; ++i) {
                            const PercentageVolumeResult out = self.update(static_cast<double>(values[i]));
                            pvo[i] = out.pvo;
                            signal[i] = out.signal;
                            histogram[i] = out.histogram;
                        }
                        return PercentageVolumeBatchResult{
                            make_array(std::move(pvo)),
                            make_array(std::move(signal)),
                            make_array(std::move(histogram)),
                        };
                    }
                    case InputDType::Float64:
                        return self.batch(nb::cast<InputArray>(volume));
                }
            }

            std::vector<double> pvo = make_record_output(records);
            std::vector<double> signal;
            std::vector<double> histogram;
            signal.reserve(pvo.capacity());
            histogram.reserve(pvo.capacity());
            for (nb::handle record : records) {
                const PercentageVolumeResult out = self.update(record_value(record, "volume", 0));
                pvo.push_back(out.pvo);
                signal.push_back(out.signal);
                histogram.push_back(out.histogram);
            }
            return PercentageVolumeBatchResult{
                make_array(std::move(pvo)),
                make_array(std::move(signal)),
                make_array(std::move(histogram)),
            };
        }, nb::arg("records"));

    nb::class_<PlusDirectionalIndicator>(m, "PlusDirectionalIndicator")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &PlusDirectionalIndicator::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(PlusDirectionalIndicator, close, high, low)
        RTTA_REPLAY3(PlusDirectionalIndicator, close, high, low)
        RTTA_BATCH3_ARRAY(PlusDirectionalIndicator, close, high, low, "close", "high", "low");

    nb::class_<PlusDirectionalMovement>(m, "PlusDirectionalMovement")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &PlusDirectionalMovement::update, nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE2(PlusDirectionalMovement, high, low)
        RTTA_REPLAY2(PlusDirectionalMovement, high, low)
        RTTA_BATCH2_ARRAY(PlusDirectionalMovement, high, low, "high", "low");

    nb::class_<ROC>(m, "ROC")
        .def(nb::init<int, bool>(), nb::arg("window"), nb::arg("fillna") = true)
        .def("update", &ROC::update, nb::arg("close"))
        RTTA_ADVANCE1(ROC, close)
        RTTA_REPLAY1(ROC, close)
        .def("batch", &ROC::batch, array_arg("close"))
        .def("batch", [](ROC &self, const FloatInputArray &close) {
            return batch_update1(self, close);
        }, array_arg("close"))
        .def("batch", [](ROC &self, nb::iterable records) {
            return batch_records_one(self, records, "close");
        }, nb::arg("records"));

    nb::class_<RateOfChangePercentage>(m, "RateOfChangePercentage")
        .def(nb::init<int, bool>(), nb::arg("window") = 10, nb::arg("fillna") = true)
        .def("update", &RateOfChangePercentage::update, nb::arg("close"))
        RTTA_ADVANCE1(RateOfChangePercentage, close)
        RTTA_REPLAY1(RateOfChangePercentage, close)
        RTTA_BATCH1_ARRAY(RateOfChangePercentage, close, "close");

    nb::class_<RateOfChangeRatio>(m, "RateOfChangeRatio")
        .def(nb::init<int, bool>(), nb::arg("window") = 10, nb::arg("fillna") = true)
        .def("update", &RateOfChangeRatio::update, nb::arg("close"))
        RTTA_ADVANCE1(RateOfChangeRatio, close)
        RTTA_REPLAY1(RateOfChangeRatio, close)
        RTTA_BATCH1_ARRAY(RateOfChangeRatio, close, "close");

    nb::class_<RateOfChangeRatio100>(m, "RateOfChangeRatio100")
        .def(nb::init<int, bool>(), nb::arg("window") = 10, nb::arg("fillna") = true)
        .def("update", &RateOfChangeRatio100::update, nb::arg("close"))
        RTTA_ADVANCE1(RateOfChangeRatio100, close)
        RTTA_REPLAY1(RateOfChangeRatio100, close)
        RTTA_BATCH1_ARRAY(RateOfChangeRatio100, close, "close");

    nb::class_<RSI>(m, "RSI")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &RSI::update, nb::arg("value"))
        RTTA_ADVANCE1(RSI, value)
        RTTA_REPLAY1(RSI, value)
        RTTA_BATCH1(RSI, value, "value");

    nb::class_<ParabolicSAR>(m, "ParabolicSAR")
        .def(nb::init<double, double>(), nb::arg("acceleration") = 0.02, nb::arg("maximum") = 0.2)
        .def("update", &ParabolicSAR::update, nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE2(ParabolicSAR, high, low)
        RTTA_REPLAY2(ParabolicSAR, high, low)
        RTTA_BATCH2(ParabolicSAR, high, low, "high", "low");

    nb::class_<SchaffTrendCycle>(m, "SchaffTrendCycle")
        .def(nb::init<int, int, int, int, int, bool>(),
             nb::arg("slow") = 50,
             nb::arg("fast") = 23,
             nb::arg("cycle") = 10,
             nb::arg("smooth1") = 3,
             nb::arg("smooth2") = 3,
             nb::arg("fillna") = true)
        .def("update", &SchaffTrendCycle::update, nb::arg("close"))
        RTTA_ADVANCE1(SchaffTrendCycle, close)
        RTTA_REPLAY1(SchaffTrendCycle, close)
        .def("batch", &SchaffTrendCycle::batch, array_arg("close"))
        .def("batch", [](SchaffTrendCycle &self, const FloatInputArray &close) {
            return batch_update1(self, close);
        }, array_arg("close"))
        .def("batch", [](SchaffTrendCycle &self, nb::iterable records) {
            return batch_records_one(self, records, "close");
        }, nb::arg("records"));

    nb::class_<Summation>(m, "Summation")
        .def(nb::init<int, bool>(), nb::arg("window"), nb::arg("fillna") = true)
        .def("update", &Summation::update, nb::arg("value"))
        RTTA_ADVANCE1(Summation, value)
        RTTA_REPLAY1(Summation, value)
        .def("batch", &Summation::batch, array_arg("input"))
        .def("batch", [](Summation &self, const FloatInputArray &input) {
            return batch_update1(self, input);
        }, array_arg("input"))
        .def("batch", [](Summation &self, nb::iterable records) {
            return batch_records_one(self, records, "input");
        }, nb::arg("records"));

    nb::class_<Stochastic>(m, "Stochastic")
        .def(nb::init<int, int, int, bool>(),
             nb::arg("fastk") = 5,
             nb::arg("slowk") = 3,
             nb::arg("slowd") = 3,
             nb::arg("fillna") = true)
        .def("update", &Stochastic::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(Stochastic, close, high, low)
        RTTA_REPLAY3(Stochastic, close, high, low)
        RTTA_FIELD3(Stochastic, slowk, close, high, low)
        RTTA_FIELD3(Stochastic, slowd, close, high, low)
        RTTA_REPLAY_OUTPUTS3(Stochastic, close, high, low, batch_stochastic)
        .def("batch", [](Stochastic &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return batch_stochastic(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](Stochastic &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return batch_stochastic(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](Stochastic &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return batch_stochastic(indicator, close, high, low);
                });
            }

            std::vector<double> slowk = make_record_output(records);
            std::vector<double> slowd;
            slowd.reserve(slowk.capacity());
            for (nb::handle record : records) {
                const StochasticResult out = self.update(
                    record_value(record, "close", 0),
                    record_value(record, "high", 1),
                    record_value(record, "low", 2));
                slowk.push_back(out.slowk);
                slowd.push_back(out.slowd);
            }
            return StochasticBatchResult{make_array(std::move(slowk)), make_array(std::move(slowd))};
        }, nb::arg("records"));

    nb::class_<FastStochastic>(m, "FastStochastic")
        .def(nb::init<int, int, bool>(), nb::arg("fastk") = 5, nb::arg("fastd") = 3, nb::arg("fillna") = true)
        .def("update", &FastStochastic::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(FastStochastic, close, high, low)
        RTTA_REPLAY3(FastStochastic, close, high, low)
        RTTA_FIELD3(FastStochastic, fastk, close, high, low)
        RTTA_FIELD3(FastStochastic, fastd, close, high, low)
        RTTA_REPLAY_OUTPUTS3(FastStochastic, close, high, low, batch_fast_stochastic)
        .def("batch", [](FastStochastic &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return batch_fast_stochastic(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](FastStochastic &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return batch_fast_stochastic(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](FastStochastic &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return batch_fast_stochastic(indicator, close, high, low);
                });
            }

            std::vector<double> fastk = make_record_output(records);
            std::vector<double> fastd;
            fastd.reserve(fastk.capacity());
            for (nb::handle record : records) {
                const FastStochasticResult out = self.update(
                    record_value(record, "close", 0),
                    record_value(record, "high", 1),
                    record_value(record, "low", 2));
                fastk.push_back(out.fastk);
                fastd.push_back(out.fastd);
            }
            return FastStochasticBatchResult{make_array(std::move(fastk)), make_array(std::move(fastd))};
        }, nb::arg("records"));

    nb::class_<High>(m, "High")
        .def(nb::init<int, bool>(), nb::arg("window"), nb::arg("fillna") = true)
        .def("update", &High::update, nb::arg("value"))
        RTTA_ADVANCE1(High, value)
        RTTA_REPLAY1(High, value)
        RTTA_BATCH1_ARRAY(High, value, "value");

    nb::class_<Low>(m, "Low")
        .def(nb::init<int, bool>(), nb::arg("window"), nb::arg("fillna") = true)
        .def("update", &Low::update, nb::arg("value"))
        RTTA_ADVANCE1(Low, value)
        RTTA_REPLAY1(Low, value)
        RTTA_BATCH1_ARRAY(Low, value, "value");

    nb::class_<StochRSI>(m, "StochRSI")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &StochRSI::update, nb::arg("value"))
        RTTA_ADVANCE1(StochRSI, value)
        RTTA_REPLAY1(StochRSI, value)
        RTTA_BATCH1(StochRSI, value, "value");

    nb::class_<T3MovingAverage>(m, "T3MovingAverage")
        .def(nb::init<double, double, bool>(), nb::arg("window") = 5.0, nb::arg("vfactor") = 0.7, nb::arg("fillna") = true)
        .def("update", &T3MovingAverage::update, nb::arg("value"))
        RTTA_ADVANCE1(T3MovingAverage, value)
        RTTA_REPLAY1(T3MovingAverage, value)
        RTTA_BATCH1_ARRAY(T3MovingAverage, value, "value");

    nb::class_<TripleEMA>(m, "TripleEMA")
        .def(nb::init<double, bool>(), nb::arg("window") = 30.0, nb::arg("fillna") = true)
        .def("update", &TripleEMA::update, nb::arg("value"))
        RTTA_ADVANCE1(TripleEMA, value)
        RTTA_REPLAY1(TripleEMA, value)
        RTTA_BATCH1(TripleEMA, value, "value");

    nb::class_<TSI>(m, "TSI")
        .def(nb::init<int, int>(), nb::arg("window_1") = 25, nb::arg("window_2") = 13)
        .def("update", &TSI::update, nb::arg("x"))
        RTTA_ADVANCE1(TSI, x)
        RTTA_REPLAY1(TSI, x)
        RTTA_BATCH1(TSI, x, "x");

    nb::class_<TrueRange>(m, "TrueRange")
        .def(nb::init<>())
        .def("update", &TrueRange::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(TrueRange, close, high, low)
        RTTA_REPLAY3(TrueRange, close, high, low)
        .def("batch", [](TrueRange &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](TrueRange &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](TrueRange &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return indicator.batch_array(close, high, low);
                });
            }
            return batch_records_three(self, records, "close", "high", "low");
        }, nb::arg("records"));

    nb::class_<TriangularMovingAverage>(m, "TriangularMovingAverage")
        .def(nb::init<int, bool>(), nb::arg("window") = 30, nb::arg("fillna") = true)
        .def("update", &TriangularMovingAverage::update, nb::arg("value"))
        RTTA_ADVANCE1(TriangularMovingAverage, value)
        RTTA_REPLAY1(TriangularMovingAverage, value)
        RTTA_BATCH1(TriangularMovingAverage, value, "value");

    nb::class_<Trix>(m, "Trix")
        .def(nb::init<double, bool>(), nb::arg("window") = 30.0, nb::arg("fillna") = true)
        .def("update", &Trix::update, nb::arg("value"))
        RTTA_ADVANCE1(Trix, value)
        RTTA_REPLAY1(Trix, value)
        RTTA_BATCH1(Trix, value, "value");

    nb::class_<TimeSeriesForecast>(m, "TimeSeriesForecast")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &TimeSeriesForecast::update, nb::arg("value"))
        RTTA_ADVANCE1(TimeSeriesForecast, value)
        RTTA_REPLAY1(TimeSeriesForecast, value)
        RTTA_BATCH1(TimeSeriesForecast, value, "value");

    nb::class_<TypicalPrice>(m, "TypicalPrice")
        .def(nb::init<>())
        .def("update", &TypicalPrice::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(TypicalPrice, close, high, low)
        RTTA_REPLAY3(TypicalPrice, close, high, low)
        RTTA_BATCH3(TypicalPrice, close, high, low, "close", "high", "low");

    nb::class_<UltimateOscillator>(m, "UltimateOscillator")
        .def(nb::init<int, int, int, bool>(),
             nb::arg("short_window") = 7,
             nb::arg("medium_window") = 14,
             nb::arg("long_window") = 28,
             nb::arg("fillna") = true)
        .def("update", &UltimateOscillator::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(UltimateOscillator, close, high, low)
        RTTA_REPLAY3(UltimateOscillator, close, high, low)
        .def("batch", [](UltimateOscillator &self, const InputArray &close, const InputArray &high, const InputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](UltimateOscillator &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return self.batch_array(close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](UltimateOscillator &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return indicator.batch_array(close, high, low);
                });
            }
            return batch_records_three(self, records, "close", "high", "low");
        }, nb::arg("records"));

    nb::class_<UlcerIndex>(m, "UlcerIndex")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &UlcerIndex::update, nb::arg("close"))
        RTTA_ADVANCE1(UlcerIndex, close)
        RTTA_REPLAY1(UlcerIndex, close)
        .def("batch", &UlcerIndex::batch, array_arg("close"))
        .def("batch", [](UlcerIndex &self, const FloatInputArray &close) {
            return batch_update1(self, close);
        }, array_arg("close"))
        .def("batch", [](UlcerIndex &self, nb::iterable records) {
            return batch_records_one(self, records, "close");
        }, nb::arg("records"));

    nb::class_<Variance>(m, "Variance")
        .def(nb::init<int, bool>(), nb::arg("window") = 5, nb::arg("fillna") = true)
        .def("update", &Variance::update, nb::arg("value"))
        RTTA_ADVANCE1(Variance, value)
        RTTA_REPLAY1(Variance, value)
        RTTA_BATCH1_ARRAY(Variance, value, "value");

    nb::class_<VolumePriceTrend>(m, "VolumePriceTrend")
        .def(nb::init<int, bool>(), nb::arg("smoothing_window") = 0, nb::arg("fillna") = true)
        .def("update", &VolumePriceTrend::update, nb::arg("close"), nb::arg("volume"))
        RTTA_ADVANCE2(VolumePriceTrend, close, volume)
        RTTA_REPLAY2(VolumePriceTrend, close, volume)
        .def("batch", &VolumePriceTrend::batch, array_arg("close"), array_arg("volume"))
        .def("batch", [](VolumePriceTrend &self, const FloatInputArray &close, const FloatInputArray &volume) {
            return batch_update2(self, close, volume);
        }, array_arg("close"), array_arg("volume"))
        .def("batch", [](VolumePriceTrend &self, nb::iterable records) {
            return batch_records_two(self, records, "close", "volume");
        }, nb::arg("records"));

    nb::class_<NegativeVolumeIndex>(m, "NegativeVolumeIndex")
        .def(nb::init<>())
        .def("update", &NegativeVolumeIndex::update, nb::arg("close"), nb::arg("volume"))
        RTTA_ADVANCE2(NegativeVolumeIndex, close, volume)
        RTTA_REPLAY2(NegativeVolumeIndex, close, volume)
        .def("batch", &NegativeVolumeIndex::batch, array_arg("close"), array_arg("volume"))
        .def("batch", [](NegativeVolumeIndex &self, const FloatInputArray &close, const FloatInputArray &volume) {
            return batch_update2(self, close, volume);
        }, array_arg("close"), array_arg("volume"))
        .def("batch", [](NegativeVolumeIndex &self, nb::iterable records) {
            return batch_records_two(self, records, "close", "volume");
        }, nb::arg("records"));

    nb::class_<AnchoredVWAP>(m, "AnchoredVWAP")
        .def(nb::init<>())
        .def("update", &AnchoredVWAP::update, nb::arg("close"), nb::arg("high"), nb::arg("low"), nb::arg("volume"), nb::arg("anchor"))
        .def("advance", [](AnchoredVWAP &self, double close, double high, double low, double volume, double anchor) {
            self.advance(close, high, low, volume, anchor);
        }, nb::arg("close"), nb::arg("high"), nb::arg("low"), nb::arg("volume"), nb::arg("anchor"))
        .def("last", &AnchoredVWAP::last)
        .def("replay_update", [](AnchoredVWAP &self, const InputArray &close, const InputArray &high, const InputArray &low, const InputArray &volume, const InputArray &anchor) {
            return self.replay_update_array(close, high, low, volume, anchor);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"), array_arg("anchor"))
        .def("replay_update", [](AnchoredVWAP &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &volume, const FloatInputArray &anchor) {
            return self.replay_update_array(close, high, low, volume, anchor);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"), array_arg("anchor"))
        .def("replay_advance", [](AnchoredVWAP &self, const InputArray &close, const InputArray &high, const InputArray &low, const InputArray &volume, const InputArray &anchor) {
            return self.replay_advance_array(close, high, low, volume, anchor);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"), array_arg("anchor"))
        .def("replay_advance", [](AnchoredVWAP &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &volume, const FloatInputArray &anchor) {
            return self.replay_advance_array(close, high, low, volume, anchor);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"), array_arg("anchor"))
        .def("batch", [](AnchoredVWAP &self, const InputArray &close, const InputArray &high, const InputArray &low, const InputArray &volume, const InputArray &anchor) {
            return self.batch_array(close, high, low, volume, anchor);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"), array_arg("anchor"))
        .def("batch", [](AnchoredVWAP &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &volume, const FloatInputArray &anchor) {
            return self.batch_array(close, high, low, volume, anchor);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"), array_arg("anchor"))
        .def("batch", [](AnchoredVWAP &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table5(self, records, "close", "high", "low", "volume", "anchor", [](auto &indicator, const auto &close, const auto &high, const auto &low, const auto &volume, const auto &anchor) {
                    return indicator.batch_array(close, high, low, volume, anchor);
                });
            }
            return batch_records_five(self, records, "close", "high", "low", "volume", "anchor");
        }, nb::arg("records"));

    nb::class_<VolumeProfile>(m, "VolumeProfile")
        .def(nb::init<int, int, double, bool>(),
             nb::arg("window") = 128,
             nb::arg("bins") = 24,
             nb::arg("value_area_percent") = 70.0,
             nb::arg("fillna") = true)
        .def("update", &VolumeProfile::update, nb::arg("close"), nb::arg("volume"))
        .def("last", &VolumeProfile::last)
        RTTA_ADVANCE2(VolumeProfile, close, volume)
        RTTA_REPLAY2(VolumeProfile, close, volume)
        RTTA_FIELD2(VolumeProfile, point_of_control, close, volume)
        RTTA_FIELD2(VolumeProfile, value_area_high, close, volume)
        RTTA_FIELD2(VolumeProfile, value_area_low, close, volume)
        .def("replay_update_outputs", [](VolumeProfile &self, const InputArray &close, const InputArray &volume) {
            return self.batch_array(close, volume);
        }, array_arg("close"), array_arg("volume"))
        .def("replay_update_outputs", [](VolumeProfile &self, const FloatInputArray &close, const FloatInputArray &volume) {
            return self.batch_array(close, volume);
        }, array_arg("close"), array_arg("volume"))
        .def("batch", [](VolumeProfile &self, const InputArray &close, const InputArray &volume) {
            return self.batch_array(close, volume);
        }, array_arg("close"), array_arg("volume"))
        .def("batch", [](VolumeProfile &self, const FloatInputArray &close, const FloatInputArray &volume) {
            return self.batch_array(close, volume);
        }, array_arg("close"), array_arg("volume"))
        .def("batch", [](VolumeProfile &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table2(self, records, "close", "volume", [](auto &indicator, const auto &close, const auto &volume) {
                    return indicator.batch_array(close, volume);
                });
            }

            std::vector<double> point_of_control = make_record_output(records);
            std::vector<double> value_area_high;
            std::vector<double> value_area_low;
            value_area_high.reserve(point_of_control.capacity());
            value_area_low.reserve(point_of_control.capacity());
            for (nb::handle record : records) {
                const VolumeProfileResult out = self.update(
                    record_value(record, "close", 0),
                    record_value(record, "volume", 1));
                point_of_control.push_back(out.point_of_control);
                value_area_high.push_back(out.value_area_high);
                value_area_low.push_back(out.value_area_low);
            }
            return VolumeProfileBatchResult{
                make_array(std::move(point_of_control)),
                make_array(std::move(value_area_high)),
                make_array(std::move(value_area_low)),
            };
        }, nb::arg("records"));

    nb::class_<VolumeWeightedAveragePrice>(m, "VolumeWeightedAveragePrice")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &VolumeWeightedAveragePrice::update, nb::arg("close"), nb::arg("high"), nb::arg("low"), nb::arg("volume"))
        RTTA_ADVANCE4(VolumeWeightedAveragePrice, close, high, low, volume)
        RTTA_REPLAY4(VolumeWeightedAveragePrice, close, high, low, volume)
        .def("batch", &VolumeWeightedAveragePrice::batch, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](VolumeWeightedAveragePrice &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low, const FloatInputArray &volume) {
            return batch_update4(self, close, high, low, volume);
        }, array_arg("close"), array_arg("high"), array_arg("low"), array_arg("volume"))
        .def("batch", [](VolumeWeightedAveragePrice &self, nb::iterable records) {
            return batch_records_four(self, records, "close", "high", "low", "volume");
        }, nb::arg("records"));

    nb::class_<VolumeWeightedMovingAverage>(m, "VolumeWeightedMovingAverage")
        .def(nb::init<int, bool>(), nb::arg("window") = 20, nb::arg("fillna") = true)
        .def("update", &VolumeWeightedMovingAverage::update, nb::arg("close"), nb::arg("volume"))
        RTTA_ADVANCE2(VolumeWeightedMovingAverage, close, volume)
        RTTA_REPLAY2(VolumeWeightedMovingAverage, close, volume)
        .def("batch", [](VolumeWeightedMovingAverage &self, const InputArray &close, const InputArray &volume) {
            return self.batch_array(close, volume);
        }, array_arg("close"), array_arg("volume"))
        .def("batch", [](VolumeWeightedMovingAverage &self, const FloatInputArray &close, const FloatInputArray &volume) {
            return self.batch_array(close, volume);
        }, array_arg("close"), array_arg("volume"))
        .def("batch", [](VolumeWeightedMovingAverage &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table2(self, records, "close", "volume", [](auto &indicator, const auto &close, const auto &volume) {
                    return indicator.batch_array(close, volume);
                });
            }
            return batch_records_two(self, records, "close", "volume");
        }, nb::arg("records"));

    nb::class_<Vortex>(m, "Vortex")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &Vortex::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(Vortex, close, high, low)
        RTTA_REPLAY3(Vortex, close, high, low)
        RTTA_FIELD3(Vortex, positive, close, high, low)
        RTTA_FIELD3(Vortex, negative, close, high, low)
        RTTA_FIELD3(Vortex, difference, close, high, low)
        RTTA_REPLAY_OUTPUTS3(Vortex, close, high, low, batch_vortex)
        .def("batch", &Vortex::batch, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](Vortex &self, const FloatInputArray &close, const FloatInputArray &high, const FloatInputArray &low) {
            return batch_vortex(self, close, high, low);
        }, array_arg("close"), array_arg("high"), array_arg("low"))
        .def("batch", [](Vortex &self, nb::iterable records) {
            if (table_has_column(records, "close")) {
                return dispatch_table3(self, records, "close", "high", "low", [](auto &indicator, const auto &close, const auto &high, const auto &low) {
                    return batch_vortex(indicator, close, high, low);
                });
            }

            std::vector<double> positive = make_record_output(records);
            std::vector<double> negative;
            std::vector<double> difference;
            negative.reserve(positive.capacity());
            difference.reserve(positive.capacity());
            for (nb::handle record : records) {
                const VortexResult out = self.update(
                    record_value(record, "close", 0),
                    record_value(record, "high", 1),
                    record_value(record, "low", 2));
                positive.push_back(out.positive);
                negative.push_back(out.negative);
                difference.push_back(out.difference);
            }
            return VortexBatchResult{
                make_array(std::move(positive)),
                make_array(std::move(negative)),
                make_array(std::move(difference)),
            };
        }, nb::arg("records"));

    nb::class_<WeightedClosePrice>(m, "WeightedClosePrice")
        .def(nb::init<>())
        .def("update", &WeightedClosePrice::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(WeightedClosePrice, close, high, low)
        RTTA_REPLAY3(WeightedClosePrice, close, high, low)
        RTTA_BATCH3(WeightedClosePrice, close, high, low, "close", "high", "low");

    nb::class_<WilliamsR>(m, "WilliamsR")
        .def(nb::init<int, bool>(), nb::arg("window") = 14, nb::arg("fillna") = true)
        .def("update", &WilliamsR::update, nb::arg("close"), nb::arg("high"), nb::arg("low"))
        RTTA_ADVANCE3(WilliamsR, close, high, low)
        RTTA_REPLAY3(WilliamsR, close, high, low)
        RTTA_BATCH3_ARRAY(WilliamsR, close, high, low, "close", "high", "low");

    nb::class_<WeightedMovingAverage>(m, "WeightedMovingAverage")
        .def(nb::init<int, bool>(), nb::arg("window") = 30, nb::arg("fillna") = true)
        .def("update", &WeightedMovingAverage::update, nb::arg("value"))
        RTTA_ADVANCE1(WeightedMovingAverage, value)
        RTTA_REPLAY1(WeightedMovingAverage, value)
        RTTA_BATCH1_ARRAY(WeightedMovingAverage, value, "value");

    nb::class_<StdDev>(m, "StdDev")
        .def(nb::init<int, bool>(), nb::arg("window"), nb::arg("fillna") = true)
        .def("update", &StdDev::update, nb::arg("value"))
        RTTA_ADVANCE1(StdDev, value)
        RTTA_REPLAY1(StdDev, value)
        RTTA_BATCH1_ARRAY(StdDev, value, "value");

    nb::class_<BollingerBands>(m, "BollingerBands")
        .def(nb::init<int, bool>(), nb::arg("window") = 20, nb::arg("fillna") = true)
        .def("update", &BollingerBands::update, nb::arg("value"))
        RTTA_ADVANCE1(BollingerBands, value)
        RTTA_REPLAY1(BollingerBands, value)
        RTTA_FIELD1(BollingerBands, middle, value)
        RTTA_FIELD1(BollingerBands, upper, value)
        RTTA_FIELD1(BollingerBands, lower, value)
        RTTA_REPLAY_OUTPUTS1(BollingerBands, value, batch_bollinger_bands)
        .def("batch", [](BollingerBands &self, const InputArray &value) {
            return batch_bollinger_bands(self, value);
        }, array_arg("value"))
        .def("batch", [](BollingerBands &self, const FloatInputArray &value) {
            return batch_bollinger_bands(self, value);
        }, array_arg("value"))
        .def("batch", [](BollingerBands &self, nb::iterable records) {
            if (table_has_column(records, "value")) {
                return dispatch_table1(self, records, "value", [](auto &indicator, const auto &value) {
                    return batch_bollinger_bands(indicator, value);
                });
            }

            std::vector<double> middle = make_record_output(records);
            std::vector<double> upper;
            std::vector<double> lower;
            upper.reserve(middle.capacity());
            lower.reserve(middle.capacity());
            for (nb::handle record : records) {
                const BollingerBandsResult out = self.update(record_value(record, "value", 0));
                middle.push_back(out.middle);
                upper.push_back(out.upper);
                lower.push_back(out.lower);
            }
            return BollingerBandsBatchResult{
                make_array(std::move(middle)),
                make_array(std::move(upper)),
                make_array(std::move(lower)),
            };
        }, nb::arg("records"));
}
