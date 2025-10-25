def get_chained_attributes(obj, attributes):
    """Getattr that allows chained attributes."""
    for attribute in attributes.split("."):
        obj = getattr(obj, attribute)
    return obj


def get_inner_most_object_from_chained_attributes(obj: object, attributes: str):
    attributes = attributes.split(".")
    for attribute in attributes[:-1]:
        obj = getattr(obj, attribute)
    return obj
