"""
Simplified VM code which works for most python programms
"""

import builtins
import dis
import types
import typing as tp
import operator
from typing import Any


class Frame:
    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None
        self.last_exception: Any = None
        self.last_byte = 0

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def jump(self, jump: int) -> None:
        self.last_byte = jump

    def jump_forward_op(self, jump: int) -> None:
        self.jump(self.last_byte + jump)

    def pop_jump_if_true_op(self, jump: int) -> None:
        val = self.pop()
        if val:
            self.jump(jump)

    def pop_jump_if_false_op(self, jump: int) -> None:
        val = self.pop()
        if not val:
            self.jump(jump)

    def jump_if_true_or_pop_op(self, jump: int) -> None:
        val = self.top()
        if val:
            self.jump(jump)
        else:
            self.pop()

    def jump_if_false_or_pop_op(self, jump: int) -> None:
        val = self.top()
        if not val:
            self.jump(jump)
        else:
            self.pop()

    def jump_absolute_op(self, jump: int) -> None:
        self.jump(jump)

    def jump_if_not_exc_match_op(self, jump: int) -> None:
        pass

    def run(self) -> tp.Any:
        instructions = list(dis.get_instructions(self.code))
        while self.last_byte // 2 < len(instructions):
            index = self.last_byte // 2
            instruction = instructions[index]
            getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
            if self.last_byte // 2 == index:
                self.last_byte += 2
            if instruction.opname.lower() == "return_value":
                return self.return_value
        return self.return_value

    def call_function_op(self, arg: int) -> None:
        arguments = self.popn(arg)
        f = self.pop()
        self.push(f(*arguments))

    def call_method_op(self, arg: int) -> None:
        pass

    def load_name_op(self, arg: str) -> None:
        if arg in self.locals:
            val = self.locals[arg]
        elif arg in self.globals:
            val = self.globals[arg]
        elif arg in self.builtins:
            val = self.builtins[arg]
        else:
            raise NameError("name '%s' is not defined" % arg)
        self.push(val)

    def load_global_op(self, arg: str) -> None:
        if arg in self.globals:
            val = self.globals[arg]
        elif arg in self.builtins:
            val = self.builtins[arg]
        else:
            raise NameError()
        self.push(val)

    def load_const_op(self, arg: tp.Any) -> None:
        self.push(arg)

    def load_fast_op(self, name: tp.Any) -> None:
        if name in self.locals:
            val = self.locals[name]
        else:
            raise UnboundLocalError(
                "local variable '%s' referenced before assignment" % name
            )
        self.push(val)

    def return_value_op(self, arg: tp.Any) -> None:
        self.return_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        self.pop()

    def make_function_op(self, arg: int) -> None:
        name = self.pop()  # the qualified name of the function (at TOS)  # noqa
        code = self.pop()  # the code associated with the function (at TOS1)
        if arg & 1 != 0:
            default_args = self.pop()
        else:
            default_args = ()
        if arg & 2 != 0:
            default_kwargs = self.pop()
        else:
            default_kwargs = {}
        # TODO: use arg to parse function defaults
        CO_VARARGS = 4
        CO_VARKEYWORDS = 8

        ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
        ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
        ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
        ERR_MISSING_POS_ARGS = 'Missing positional arguments'
        ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
        ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'

        def func(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            # TODO: parse input arguments using code attributes such as co_argcount
            flags: int = code.co_flags
            args_used: bool = (flags & CO_VARARGS) != 0
            kwargs_used: bool = (flags & CO_VARKEYWORDS) != 0

            pos_names: Any = code.co_varnames[:code.co_argcount]
            pos_names_dict: Any = dict(zip(pos_names, [False for _ in pos_names]))
            assigned_pos: Any = 0
            for kwarg in kwargs:
                if kwarg in pos_names_dict:
                    pos_names_dict[kwarg] = True
                    assigned_pos += 1

            pos_where_defs_start: Any = code.co_argcount - len(
                [] if func.__defaults__ is None else func.__defaults__)
            non_default_pos_names: Any = code.co_varnames[:pos_where_defs_start]
            non_default_pos_names_not_initialized_with_kwargs: Any = len(non_default_pos_names)

            if len(args) == 0 and len(code.co_varnames[:pos_where_defs_start]) < len(
                    kwargs) and args_used and kwargs_used:
                raise TypeError(ERR_MISSING_POS_ARGS)
            for kw in kwargs:
                if kw in non_default_pos_names:
                    non_default_pos_names_not_initialized_with_kwargs -= 1

            if non_default_pos_names_not_initialized_with_kwargs > len(args):
                raise TypeError(ERR_MISSING_POS_ARGS)

            if args_used:
                pass
            else:
                if len(args) > code.co_argcount:
                    raise TypeError(ERR_TOO_MANY_POS_ARGS)

            num_kw_def: int = 0 if func.__kwdefaults__ is None else len(func.__kwdefaults__)
            if code.co_kwonlyargcount - num_kw_def > len(kwargs):
                raise TypeError(ERR_MISSING_KWONLY_ARGS)

            default_pos_names: Any = code.co_varnames[len(args):code.co_argcount]
            default_pos_values: Any = default_args
            defaults_pos: Any = dict(zip(default_pos_names, default_pos_values))
            defaults_kw: Any = default_kwargs
            binded_defaults: Any = defaults_pos | defaults_kw

            args_names: Any = code.co_varnames[:code.co_argcount]
            args_values: Any = args
            binded_args: Any = dict(zip(args_names, args_values))

            if kwargs_used:
                varnames_posonly: Any = code.co_varnames[:code.co_posonlyargcount]
                for posonly_var in varnames_posonly:
                    if posonly_var in kwargs and posonly_var not in binded_args:
                        raise TypeError(ERR_POSONLY_PASSED_AS_KW)
            else:
                varnames_posonly = code.co_varnames[:code.co_posonlyargcount]
                for posonly_var in varnames_posonly:
                    if posonly_var in kwargs:
                        raise TypeError(ERR_POSONLY_PASSED_AS_KW)

            binded_kwargs = {}
            for kw in kwargs:
                binded_kwargs[kw] = kwargs[kw]

            res = {}
            for arg in code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]:
                if arg in binded_defaults:
                    res[arg] = binded_defaults[arg]
                if arg in binded_args:
                    res[arg] = binded_args[arg]
                if arg in binded_kwargs:
                    if kwargs_used:
                        if arg in binded_args:
                            pass
                        else:
                            res[arg] = binded_kwargs[arg]
                    else:
                        res[arg] = binded_kwargs[arg]
            if args_used:
                args_name: Any = code.co_varnames[code.co_argcount + code.co_kwonlyargcount]
                res[args_name] = tuple(args[len(binded_args):])

            if kwargs_used:
                kwargs_dict: Any = {}
                for kwarg in kwargs:
                    if kwarg not in res:
                        kwargs_dict[kwarg] = kwargs[kwarg]
                    if kwarg in binded_kwargs and kwarg in binded_args:
                        kwargs_dict[kwarg] = kwargs[kwarg]
                if args_used:
                    kwargs_name = code.co_varnames[
                        code.co_argcount + code.co_kwonlyargcount + 1]
                else:
                    kwargs_name = code.co_varnames[code.co_argcount + code.co_kwonlyargcount]
                res[kwargs_name] = kwargs_dict

            if kwargs_used:
                assigned_varnames_kw: Any = code.co_varnames[code.co_posonlyargcount: len(args)]
                for kw_arg in kwargs:
                    if kw_arg in assigned_varnames_kw and kw_arg in binded_args:
                        raise TypeError(ERR_MULT_VALUES_FOR_ARG)
                        pass
            else:
                if not args_used:
                    assigned_varnames_kw = code.co_varnames[code.co_posonlyargcount: len(args)]
                    for kw_arg in kwargs:
                        if kw_arg in assigned_varnames_kw:
                            raise TypeError(ERR_MULT_VALUES_FOR_ARG)
                else:
                    pass

            parsed_args: dict[str, tp.Any] = res
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(func)

    def call_function_kw_op(self, arg: tp.Any) -> None:
        kwargs_names = self.pop()
        kwargs = self.popn(len(kwargs_names))
        args = self.popn(arg - len(kwargs_names))
        kwargs = dict(zip(kwargs_names, kwargs))
        f = self.pop()
        self.push(f(*args, **kwargs))

    def store_name_op(self, arg: str) -> None:
        const = self.pop()
        self.locals[arg] = const

    def store_global_op(self, name: str) -> None:
        self.globals[name] = self.pop()

    def store_fast_op(self, name: str) -> None:
        self.locals[name] = self.pop()

    def store_subscr_op(self, name: str) -> None:
        val, obj, subscr = self.popn(3)
        obj[subscr] = val

    def store_attr_op(self, name: str) -> None:
        val, obj = self.popn(2)
        setattr(obj, name, val)

    def delete_name_op(self, name: str) -> None:
        del self.locals[name]

    def delete_fast_op(self, name: str) -> None:
        del self.locals[name]

    def delete_global_op(self, name: str) -> None:
        del self.globals[name]

    def delete_subscr_op(self, arg: Any) -> None:
        obj, subscr = self.popn(2)
        del obj[subscr]

    def dup_top_op(self, arg: Any) -> None:
        self.push(self.top())

    def dup_top_two_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        self.push(a, b, a, b)

    def rot_two_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        self.push(b, a)

    def rot_three_op(self, arg: Any) -> None:
        a, b, c = self.popn(3)
        self.push(c, a, b)

    def rot_four_op(self, arg: Any) -> None:
        a, b, c, d = self.popn(4)
        self.push(d, a, b, c)

    # arithmetics
    def binary_add_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = a + b
        self.push(val)

    def binary_and_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.and_(a, b)
        self.push(val)

    def binary_floor_divide_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.floordiv(a, b)
        self.push(val)

    def binary_lshift_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.lshift(a, b)
        self.push(val)

    def floor_divide_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.floordiv(a, b)
        self.push(val)

    def binary_matrix_multiply_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.matmul(a, b)
        self.push(val)

    def binary_modulo_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.mod(a, b)
        self.push(val)

    def binary_multiply_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.mul(a, b)
        self.push(val)

    def binary_or_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.or_(a, b)
        self.push(val)

    def binary_power_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.pow(a, b)
        self.push(val)

    def binary_rshift_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.rshift(a, b)
        self.push(val)

    def binary_subtract_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.sub(a, b)
        self.push(val)

    def binary_true_divide_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.truediv(a, b)
        self.push(val)

    def binary_xor_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        val = operator.xor(a, b)
        self.push(val)

    def binary_subscr_op(self, arg: Any) -> None:
        a, b = self.popn(2)
        b = a[b]
        self.push(b)

    # inplace

    def inplace_add_op(self, op: str) -> None:
        a, b = self.popn(2)
        a += b
        self.push(a)

    def inplace_and_op(self, op: str) -> None:
        a, b = self.popn(2)
        a &= b
        self.push(a)

    def inplace_floor_divide_op(self, op: str) -> None:
        a, b = self.popn(2)
        a = operator.floordiv(a, b)
        self.push(a)

    def inplace_lshift_op(self, op: str) -> None:
        a, b = self.popn(2)
        a <<= b
        self.push(a)

    def inplace_modulo_op(self, op: str) -> None:
        a, b = self.popn(2)
        a %= b
        self.push(a)

    def inplace_multiply_op(self, op: str) -> None:
        a, b = self.popn(2)
        a *= b
        self.push(a)

    def inplace_or_op(self, op: str) -> None:
        a, b = self.popn(2)
        a |= b
        self.push(a)

    def inplace_power_op(self, op: str) -> None:
        a, b = self.popn(2)
        a **= b
        self.push(a)

    def inplace_rshift_op(self, op: str) -> None:
        a, b = self.popn(2)
        a >>= b
        self.push(a)

    def inplace_subtract_op(self, op: str) -> None:
        a, b = self.popn(2)
        a -= b
        self.push(a)

    def inplace_true_divide_op(self, op: str) -> None:
        a, b = self.popn(2)
        a = operator.truediv(a, b)
        self.push(a)

    def inplace_xor_op(self, op: str) -> None:
        a, b = self.popn(2)
        a ^= b
        self.push(a)

    def inplace_matrix_multiply(self, op: str) -> None:
        a, b = self.popn(2)
        a = operator.matmul(a, b)
        self.push(a)

    COMPARE_OPERATORS = {
        '<': operator.lt,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '>=': operator.ge,
        'in': lambda x, y: x in y,
        'not in': lambda x, y: x not in y,
        'is': lambda x, y: x is y,
        'is not': lambda x, y: x is not y,
        'None': lambda x, y: issubclass(x, Exception) and issubclass(x, y),
    }

    def compare_op_op(self, opname: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(self.COMPARE_OPERATORS[opname](a, b))

    def unary_invert_op(self, op: tp.Any) -> None:
        a: tp.Any = self.pop()
        self.push(operator.invert(a))

    def unary_positive_op(self, op: tp.Any) -> None:
        a: tp.Any = self.pop()
        self.push(operator.pos(a))

    def unary_negative_op(self, op: tp.Any) -> None:
        a: tp.Any = self.pop()
        self.push(operator.neg(a))

    def unary_not_op(self, op: tp.Any) -> None:
        a: tp.Any = self.pop()
        self.push(operator.not_(a))

    def contains_op_op(self, invert: tp.Any) -> None:
        a, b = self.popn(2)
        if invert == 0:
            self.push(a in b)
        else:
            self.push(a not in b)

    def is_op_op(self, invert: Any) -> None:
        a, b = self.popn(2)
        if invert == 0:
            self.push(a is b)
        else:
            self.push(a is not b)

    def build_list_op(self, arg: int) -> None:
        lst = self.popn(arg)
        self.push(lst)

    def build_slice_op(self, arg: int) -> None:
        if arg == 2:
            a, b = self.popn(2)
            self.push(slice(a, b))
        elif arg == 3:
            a, b, c = self.popn(3)
            self.push(slice(a, b, c))

    def list_extend_op(self, i: int) -> None:
        list.extend(self.data_stack[-i - 1], self.pop())

    def list_append_op(self, i: int) -> None:
        list.append(self.data_stack[-i - 1], self.pop())

    def list_to_tuple_op(self, arg: int) -> None:
        a = self.pop()
        self.push(tuple(a))

    def build_map_op(self, count: int) -> None:
        self.push({})
    def build_const_key_map_op(self, count: int) -> None:
        keys = self.pop()
        values = self.popn(count)
        res = dict(zip(keys, values))
        self.push(res)
        pass

    def dict_update_op(self, i: int) -> None:
        dict.update(self.data_stack[-i - 1], self.pop())

    def build_set_op(self, n: int) -> None:
        st = set(self.popn(n))
        self.push(st)

    def set_update_op(self, i: int) -> None:
        set.update(self.data_stack[-i - 1], self.pop())

    def set_add_op(self, i: int) -> None:
        set.add(self.data_stack[-i - 1], self.pop())

    # def build_slice_op(self, n: int) -> None:
    # lst = self.popn(n)
    # self.push(lst)

    def build_tuple_op(self, n: int) -> None:
        lst = self.popn(n)
        self.push(lst)

    def build_string_op(self, n: int) -> None:
        res = str()
        lst = self.popn(n)
        for st in lst:
            res += st
        self.push(res)

    def unpack_sequence_op(self, arg: int) -> None:
        seq = self.pop()
        for x in reversed(seq):
            self.push(x)

    def for_iter_op(self, jump: Any) -> None:
        iterobj: Any = self.top()
        try:
            v: Any = next(iterobj)
            self.push(v)
        except StopIteration:
            self.pop()
            self.jump(jump)

    def get_iter_op(self, arg: Any) -> None:
        self.push(iter(self.pop()))

    ##
    def load_assertion_error_op(self, arg: str) -> None:
        self.push(AssertionError)

    def load_method_op(self, namei: str) -> None:
        obj = self.pop()
        if getattr(obj, namei) is not None:
            self.push(obj)
            self.push(getattr(obj, namei))
        else:
            self.push(None)
            # self.push

    def load_build_class_op(self, namei: str) -> None:
        self.push(__build_class__)

    def import_name_op(self, name: str) -> None:
        namespace, from_ = self.popn(2)
        self.push(
            __import__(name, self.globals, self.locals, from_, namespace)
        )

    def import_from_op(self, name: str) -> None:
        mod = self.top()
        self.push(getattr(mod, name))

    def import_star_op(self, name: str) -> None:
        mod = self.pop()
        for attr in dir(mod):
            if attr[0] != '_':
                self.locals[attr] = getattr(mod, attr)

    def raise_varargs_op(self, argc: tp.Any) -> str:
        exctype: Any | None = None
        val: Any | None = None
        tb: Any | None = None
        if argc == 0:
            exctype, val, tb = self.last_exception
        elif argc == 1:
            exctype = self.pop()
        elif argc == 2:
            val = self.pop()
            exctype = self.pop()
        elif argc == 3:
            tb = self.pop()
            val = self.pop()
            exctype = self.pop()

        if isinstance(exctype, BaseException):
            val = exctype
            exctype = type(val)

        self.last_exception = (exctype, val, tb)

        if tb:
            return 'reraise'
        else:
            return 'exception'

    def load_attr_op(self, attr: tp.Any) -> None:
        obj: tp.Any = self.pop()
        val: tp.Any = getattr(obj, attr)
        self.push(val)

    def delete_attr_op(self, name: tp.Any) -> None:
        obj: tp.Any = self.pop()
        delattr(obj, name)

    def nop_op(self, arg: int) -> None:
        pass


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
