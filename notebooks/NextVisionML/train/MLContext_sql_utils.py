from ..util import create_object, get_next_ID_for_Table, update_object_attributes
from ..util import update_object_attributes

def load_init_parameters(mlContext):
    # Get table references
    train_process_table = mlContext.Base.classes["train_process"]
    train_process_init_parameter_table = mlContext.Base.classes["train_process_init_parameter"]
    # Load initial parameters - use latest parameters if none were explicitly given
    count_process = mlContext.session.query(train_process_table.id).count()
    count_paras = mlContext.session.query(train_process_init_parameter_table.id).count()

    # Fetch the latest init_parameters safely
    mlContext.init_parameters = mlContext.session.query(train_process_init_parameter_table).order_by(train_process_init_parameter_table.id.desc()).first()

    # Clone and add init_parameters if counts are equal
    if count_process == count_paras and mlContext.init_parameters is not None:
        init_parameters_clone = train_process_init_parameter_table()
        for column in train_process_init_parameter_table.__table__.columns:
            if column.name != "id":
                setattr(init_parameters_clone, column.name, getattr(mlContext.init_parameters, column.name))
        mlContext.session.add(init_parameters_clone)


def create_train_process_objects(mlContext):
    pass

def create_train_iteration_objects(mlContext, i):
    mlContext.iter_objs[i]["hyperparameter"] = create_object(mlContext.context, "hyperparameter",
                                id = get_next_ID_for_Table(mlContext.context, "hyperparameter"))

    mlContext.iter_objs[i]["train_process_iteration"]  = create_object(mlContext.context, "train_process_iteration",
                                            id = get_next_ID_for_Table(mlContext.context, "train_process_iteration"),
                                            train_process_id = mlContext.train_process.id,
                                                hyperparameter_id = mlContext.iter_objs[i]["hyperparameter"].id)    

    mlContext.iter_objs[i]["train_process_iteration_compute_result"]  = create_object(mlContext.context, "train_process_iteration_compute_result",
                                                            id = get_next_ID_for_Table(mlContext.context, "train_process_iteration_compute_result"),
                                                            train_process_iteration_id = mlContext.iter_objs[i]["train_process_iteration"].id)
    
    mlContext.iter_objs[i]["model"] = create_object(mlContext.context, "model",
                                                            id = get_next_ID_for_Table(mlContext.context, "model"),
                                                            train_process_iteration_id = mlContext.iter_objs[i]["train_process_iteration"].id) 
    mlContext.session.commit()
    
    mlContext.iter_objs[i]["model_score"]  = create_object(mlContext.context, "model_score",
                                                id = get_next_ID_for_Table(mlContext.context, "model_score"),
                                                model_id = mlContext.iter_objs[i]["model"].id)
    mlContext.session.commit()
    mlContext.session.commit()
    
def upload_prediction(row, mlContext, i):
    mlContext.iter_objs[i]["prediciions_categorical"]  = create_object(mlContext.context, "prediciions_categorical",
                                                            id = get_next_ID_for_Table(mlContext.context, "prediciions_categorical"),
                                                            datapoint_train_process_junction_id = mlContext.test_db_indexes.loc[row.name, "datapoint_id"],
                                                            model_id =  mlContext.iter_objs[i]["model"].id,
                                                            pred = row["eval_predict"])