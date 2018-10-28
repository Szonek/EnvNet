import inspect
class ErrorHandler:
    @classmethod
    def __get_called_method(cls):
        current_frame = inspect.currentframe()
        called_frame = inspect.getouterframes(current_frame, 4)
        return "method: " + called_frame[3][3] + " in class: " + called_frame[3][4][0][6: -2]

    @classmethod
    def __is_instance(cls, obj, type):
        if not isinstance(obj, type):
            raise Exception("Object is not a: ", type, "Called function: ", cls.__get_called_method())
        return True

    @classmethod
    def raise_error(cls, message):
        raise Exception(message, cls.__get_called_method())

    @classmethod
    def is_string(cls, obj):
        return cls.__is_instance(obj, str)

    @classmethod
    def is_dict(cls, obj):
        return cls.__is_instance(obj, dict)

    @classmethod
    def is_list(cls, obj):
        return cls.__is_instance(obj, list)

    @classmethod
    def is_type_generic(cls, obj, generic_type):
        return cls.__is_instance(obj, generic_type)