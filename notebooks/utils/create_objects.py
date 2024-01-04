def create_objects(context, table_name, object_data_list):
    """
    Create multiple objects and add them to the database in a batch.

    :param context: The database context.
    :param table_name: The name of the table where objects will be created.
    :param object_data_list: A list of dictionaries, where each dictionary contains the attributes for one object.
    :return: A list of created objects.
    """
    mapped_class = context["base"].classes[table_name]
    objects = []

    for data in object_data_list:
        obj = mapped_class()
        for key, value in data.items():
            setattr(obj, key, value)
        context["session"].add(obj)
        objects.append(obj)

    return objects