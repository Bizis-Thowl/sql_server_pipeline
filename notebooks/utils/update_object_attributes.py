def update_object_attributes(entity, attributes):
    """
    Update the attributes of a given SQLAlchemy entity.

    Args:
        entity: The SQLAlchemy entity to be updated.
        attributes: A dictionary of attribute names and their new values.
    """
    for attr, value in attributes.items():
        setattr(entity, attr, value)
    return entity