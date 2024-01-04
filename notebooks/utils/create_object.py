def create_object(context, table_name, with_commit=False, **kwargs):
    mapped_class = context["base"].classes[table_name]
    obj = mapped_class()
    for key, value in kwargs.items():
        setattr(obj, key, value)
    context["session"].add(obj)
    if (with_commit):
        context["session"].commit()
    return obj