# conftest.py
# def pytest_collection_modifyitems(session, config, items):
#     for item in items:
#         if hasattr(item, "cls") and item.cls is not None:
#             mod = item.cls.__module__  # e.g., "plugins.nnx.Test_conv"
#             mod_path = mod.replace(".", "/")  # e.g., "plugins/nnx/Test_conv"
#             # Update nodeid to reflect the new hierarchy.
#             item._nodeid = f"{mod_path}/{item.name}"  # No "tests/" prefix
