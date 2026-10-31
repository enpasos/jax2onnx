# tests/extra_tests/converter/test_python_annotations.py

"""The private annotation adapter also runs as a stdlib-only native owner.

Run this file directly to load the exact compatibility module by path without
importing the converter or a framework. Pytest uses the normal package import.
"""

import __future__
import gc
import importlib.util
from pathlib import Path
import sys
from types import FunctionType
import unittest
from unittest.mock import patch
import weakref


if __name__ == "__main__":
    _path = (
        Path(__file__).resolve().parents[3] / "jax2onnx/_compat/python_annotations.py"
    )
    _spec = importlib.util.spec_from_file_location("_python_annotations_owner", _path)
    assert _spec is not None and _spec.loader is not None
    annotations = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = annotations
    _spec.loader.exec_module(annotations)
else:
    from jax2onnx._compat import python_annotations as annotations


def _prepare(fn):
    state = annotations.AnnotationState.capture(fn)
    protected = (fn.__code__, fn.__defaults__, fn.__kwdefaults__, fn.__closure__)

    def check_metadata():
        actual = (fn.__code__, fn.__defaults__, fn.__kwdefaults__, fn.__closure__)
        if any(
            left is not right for left, right in zip(actual, protected, strict=True)
        ):
            raise annotations.AnnotationStateError(
                "protected function metadata changed"
            )

    prepared = FunctionType(
        fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    prepared.__kwdefaults__ = fn.__kwdefaults__
    peer = state.attach(prepared, check_metadata=check_metadata)
    return prepared, state, peer


class TestPythonAnnotations(unittest.TestCase):
    def test_callback_free_capture_and_checks_then_shared_explicit_value(self):
        calls = []

        def annotation():
            calls.append("annotation")
            return object

        def original(x: annotation()) -> annotation():
            return x

        if sys.version_info < (3, 14):
            original.__annotations__  # Normalize the retained eager legacy path.
        original.__dict__  # Establish the already inspected function dictionary.
        before = tuple(gc.get_referents(original))
        count = len(calls)
        first, state, first_state = _prepare(original)
        second, second_original_state, second_state = _prepare(original)
        self.assertIs(state, second_original_state)
        for current in (state, first_state, second_state):
            current.assert_current()
        self.assertEqual(len(calls), count)
        self.assertTrue(annotations._same(tuple(gc.get_referents(original)), before))
        shared = first.__annotations__
        self.assertIs(shared, original.__annotations__)
        self.assertIs(second.__annotations__, shared)
        self.assertEqual(len(calls), count + (2 if sys.version_info >= (3, 14) else 0))
        for current in (state, first_state, second_state):
            current.assert_current()
        shared["return"] = int
        for current in (state, first_state, second_state):
            with self.assertRaises(annotations.AnnotationStateError):
                current.assert_current()

    def test_cached_dict_subclass_aliases_preserve_native_bindings(self):
        class HostileDict(dict):
            def items(self):
                raise AssertionError("overridden items executed")

            def __iter__(self):
                raise AssertionError("overridden iteration executed")

            def __len__(self):
                raise AssertionError("overridden length executed")

        namespace = HostileDict(__name__="alias_owner")
        namespace["__builtins__"] = namespace
        exec("def original(x, *, retained=None):\n    return x\n", namespace)
        original = namespace["original"]
        original.__dict__ = namespace
        original.__kwdefaults__ = namespace
        original.__annotations__ = namespace
        prepared, state, peer = _prepare(original)
        state.assert_current()
        peer.assert_current()
        self.assertIs(prepared.__annotations__, namespace)
        self.assertIs(original.__annotations__, namespace)
        state.assert_current()
        peer.assert_current()
        namespace["changed"] = object()
        for current in (state, peer):
            with self.assertRaises(annotations.AnnotationStateError):
                current.assert_current()

    def test_mapping_replacement_order_and_value_changes_stale(self):
        for mutation in ("replacement", "order", "key", "value"):

            def original(x):
                return x

            original.__annotations__ = {"x": int, "return": str}
            prepared, state, peer = _prepare(original)
            mapping = prepared.__annotations__
            if mutation == "replacement":
                original.__annotations__ = dict(mapping)
            elif mutation == "order":
                value = mapping.pop("x")
                mapping["x"] = value
            elif mutation == "key":
                mapping["other"] = mapping.pop("x")
            else:
                mapping["x"] = float
            with self.assertRaises(annotations.AnnotationStateError):
                state.assert_current()
            if mutation != "replacement":
                with self.assertRaises(annotations.AnnotationStateError):
                    peer.assert_current()

    def test_original_side_first_read_stales_and_fresh_preparation_is_allowed(self):
        for mutate in (False, True):

            def original(x: int) -> str:
                return x

            prepared, state, peer = _prepare(original)
            mapping = original.__annotations__
            if mutate:
                mapping["return"] = float
            if sys.version_info >= (3, 14):
                with self.assertRaisesRegex(
                    annotations.AnnotationStateError, "fresh preparation"
                ):
                    state.assert_current()
                with self.assertRaises(annotations.AnnotationStateError):
                    prepared.__annotations__
            elif mutate:
                with self.assertRaises(annotations.AnnotationStateError):
                    state.assert_current()
            else:
                state.assert_current()
                peer.assert_current()
            fresh, fresh_state, fresh_peer = _prepare(original)
            self.assertIs(fresh.__annotations__, mapping)
            fresh_state.assert_current()
            fresh_peer.assert_current()

    def test_direct_value_read_snapshots_before_return_and_does_not_repair_peers(self):
        def original(x: int) -> str:
            return x

        first, state, first_state = _prepare(original)
        second, _, second_state = _prepare(original)
        if sys.version_info >= (3, 14):
            mapping = first.__annotate__(1)
            state.assert_current()
            first_state.assert_current()
            second_state.assert_current()
            self.assertIs(first.__annotations__, mapping)
            self.assertIs(second.__annotations__, mapping)
            first_state.assert_current()
            del first.__annotations__
            with self.assertRaises(annotations.AnnotationStateError):
                first_state.assert_current()
        else:
            mapping = first.__annotations__
            self.assertIs(second.__annotations__, mapping)
        mapping["return"] = float
        with self.assertRaises(annotations.AnnotationStateError):
            state.assert_current()

        def another(x: int):
            return x

        first, state, _ = _prepare(another)
        second, _, second_state = _prepare(another)
        if sys.version_info >= (3, 14):
            second.__annotations__ = {"changed": int}
            self.assertIs(first.__annotations__, another.__annotations__)
            state.assert_current()
            with self.assertRaises(annotations.AnnotationStateError):
                second_state.assert_current()
        else:
            self.assertIs(first.__annotations__, second.__annotations__)
            state.assert_current()

    def test_raising_annotations_retry_without_warming_or_advancing(self):
        calls = []

        def annotation():
            calls.append("annotation")
            if sys.version_info >= (3, 14):
                raise LookupError("retained annotation exception")
            return int

        def original(x: annotation()):
            return x

        prepared, state, peer = _prepare(original)
        count = len(calls)
        for attempt in range(2):
            if sys.version_info >= (3, 14):
                with self.assertRaisesRegex(
                    LookupError, "retained annotation exception"
                ):
                    prepared.__annotations__
            else:
                self.assertIs(prepared.__annotations__, original.__annotations__)
            state.assert_current()
            peer.assert_current()
            self.assertEqual(
                len(calls), count + (attempt + 1 if sys.version_info >= (3, 14) else 0)
            )

    def test_optional_slots_empty_annotations_and_traversal_fail_closed(self):
        def original(x):
            return x

        original.__module__ = None
        original.__doc__ = None
        if sys.version_info >= (3, 14):
            original.__type_params__ = ()
            original.__annotate__ = None
        prepared, state, peer = _prepare(original)
        state.assert_current()
        peer.assert_current()
        self.assertEqual(prepared.__annotations__, {})
        state.assert_current()
        peer.assert_current()
        del original.__module__
        del original.__doc__
        fresh, fresh_state, fresh_peer = _prepare(original)
        self.assertEqual(fresh.__annotations__, {})
        fresh_state.assert_current()
        fresh_peer.assert_current()
        if sys.version_info >= (3, 14):
            referents = gc.get_referents(original)
            with patch.object(
                annotations.gc, "get_referents", return_value=referents[::-1]
            ):
                with self.assertRaises(annotations.AnnotationStateError):
                    annotations.AnnotationState.capture(original)

    def test_annotationlib_public_formats_keep_original_thunk_fallback(self):
        if sys.version_info >= (3, 14):
            import annotationlib

            for expression in ("int", "UnknownAnnotation", "1 / 0"):
                namespace = {"__name__": "format_owner"}
                exec(
                    "def original(x: " + expression + ") -> list[int]:\n    return x\n",
                    namespace,
                )
                original = namespace["original"]
                prepared, state, peer = _prepare(original)
                for format in (
                    annotationlib.Format.STRING,
                    annotationlib.Format.FORWARDREF,
                    annotationlib.Format.VALUE,
                ):
                    try:
                        actual = annotationlib.get_annotations(prepared, format=format)
                    except Exception as actual_error:
                        with self.assertRaises(type(actual_error)):
                            annotationlib.get_annotations(original, format=format)
                    else:
                        expected = annotationlib.get_annotations(
                            original, format=format
                        )
                        self.assertEqual(
                            {key: repr(value) for key, value in actual.items()},
                            {key: repr(value) for key, value in expected.items()},
                        )
                    state.assert_current()
                    peer.assert_current()
        else:

            def original(x: int) -> list[int]:
                return x

            prepared, state, peer = _prepare(original)
            self.assertIs(prepared.__annotations__, original.__annotations__)
            state.assert_current()
            peer.assert_current()

    def test_annotator_replacement_code_defaults_and_closure_changes_stale(self):
        if sys.version_info >= (3, 14):
            for mutation in (
                "replace",
                "none",
                "code",
                "defaults",
                "kwdefaults",
                "cell",
            ):
                retained = object()

                def annotate(format, value=None, *, flag=None):
                    return {"x": retained}

                def original(x):
                    return x

                original.__annotate__ = annotate
                prepared, state, peer = _prepare(original)
                if mutation == "replace":
                    original.__annotate__ = lambda format: {}
                elif mutation == "none":
                    original.__annotate__ = None
                elif mutation == "code":

                    def alternate(format, value=None, *, flag=None):
                        return {"return": retained}

                    annotate.__code__ = alternate.__code__
                elif mutation == "defaults":
                    annotate.__defaults__ = (int,)
                elif mutation == "kwdefaults":
                    annotate.__kwdefaults__["flag"] = int
                else:
                    annotate.__closure__[0].cell_contents = int
                with self.assertRaises(annotations.AnnotationStateError):
                    state.assert_current()
                with self.assertRaises(annotations.AnnotationStateError):
                    prepared.__annotations__
                peer.assert_current()
        else:

            def original(x: int):
                return x

            prepared, state, peer = _prepare(original)
            self.assertIs(prepared.__annotations__, original.__annotations__)
            state.assert_current()
            peer.assert_current()

    def test_annotationlib_retains_unresolved_empty_annotation_cells(self):
        def factory():
            captured = int

            def original(x: captured) -> list[captured]:
                return x

            del captured
            return original

        original = factory()
        prepared, state, peer = _prepare(original)
        self.assertIsNone(original.__closure__)
        state.assert_current()
        peer.assert_current()
        if sys.version_info >= (3, 14):
            import annotationlib

            for format in (
                annotationlib.Format.STRING,
                annotationlib.Format.FORWARDREF,
            ):
                actual = annotationlib.get_annotations(prepared, format=format)
                expected = annotationlib.get_annotations(original, format=format)
                self.assertEqual(
                    {key: repr(value) for key, value in actual.items()},
                    {key: repr(value) for key, value in expected.items()},
                )
                state.assert_current()
                peer.assert_current()
            with self.assertRaises(NameError):
                prepared.__annotations__
            state.assert_current()
            peer.assert_current()
        else:
            self.assertIs(prepared.__annotations__, original.__annotations__)

    def test_callback_metadata_changes_and_reentrancy_do_not_advance_baseline(self):
        if sys.version_info >= (3, 14):
            for mutation in ("defaults", "annotator-before-error", "reentrant"):
                holder = []

                def original(x, value=None):
                    return x

                def annotate(format):
                    if mutation == "defaults":
                        original.__defaults__ = (int,)
                        return {"x": int}
                    if mutation == "annotator-before-error":
                        original.__annotate__ = None
                        raise LookupError("changed annotation state")
                    return holder[0].__annotations__

                original.__annotate__ = annotate
                prepared, state, peer = _prepare(original)
                holder.append(prepared)
                with self.assertRaises(annotations.AnnotationStateError):
                    prepared.__annotations__
                if mutation == "reentrant":
                    state.assert_current()
                    peer.assert_current()
                else:
                    with self.assertRaises(annotations.AnnotationStateError):
                        state.assert_current()
        else:

            def original(x):
                return x

            prepared, state, peer = _prepare(original)
            self.assertEqual(prepared.__annotations__, {})
            state.assert_current()
            peer.assert_current()

    def test_future_annotations_generic_and_custom_owner_boundaries(self):
        namespace = {"__name__": "future_owner"}
        exec(
            compile(
                "def original(x: UnknownAnnotation) -> int:\n    return x\n",
                "<future-owner>",
                "exec",
                flags=__future__.annotations.compiler_flag,
                dont_inherit=True,
            ),
            namespace,
        )
        original = namespace["original"]
        prepared, state, peer = _prepare(original)
        state.assert_current()
        peer.assert_current()
        self.assertIs(prepared.__annotations__, original.__annotations__)
        self.assertEqual(original.__annotations__["x"], "UnknownAnnotation")
        state.assert_current()
        peer.assert_current()
        if sys.version_info >= (3, 14):
            exec("def generic[T](x: T) -> T:\n    return x\n", namespace)
            with self.assertRaisesRegex(annotations.AnnotationStateError, "generic"):
                annotations.AnnotationState.capture(namespace["generic"])

            class Annotator:
                def __call__(self, format):
                    raise AssertionError("unsupported owner executed")

            original.__annotate__ = Annotator()
            with self.assertRaisesRegex(
                annotations.AnnotationStateError, "non-function"
            ):
                annotations.AnnotationState.capture(original)

    def test_unknown_raw_slots_never_invoke_class_hooks_or_getters(self):
        class Hostile:
            @property
            def __class__(self):
                raise AssertionError("hostile __class__ hook executed")

        def original(x):
            return x

        original.__dict__
        if sys.version_info >= (3, 14):
            referents = gc.get_referents(original)
            for unexpected in (Hostile(), ("x", int)):
                malformed = [*referents[:-1], unexpected, referents[-1]]
                with patch.object(
                    annotations.gc, "get_referents", return_value=malformed
                ):
                    with self.assertRaises(annotations.AnnotationStateError):
                        annotations.AnnotationState.capture(original)
        else:
            prepared, state, peer = _prepare(original)
            self.assertEqual(prepared.__annotations__, {})
            state.assert_current()
            peer.assert_current()

    def test_weak_coordinator_releases_functions_and_peers(self):
        def build():
            def original(x: int):
                return x

            prepared, state, peer = _prepare(original)
            return (
                weakref.ref(original),
                weakref.ref(prepared),
                weakref.ref(state),
                weakref.ref(peer),
            )

        references = build()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))


if __name__ == "__main__":
    unittest.main()
