from .load_context import load_context
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from ..util import create_object, get_next_ID_for_Table, update_object_attributes
from ..util import update_object_attributes
from .MLContext_sql_utils import create_train_iteration_objects, load_init_parameters
from .train_utils import sample_snapshot
from .MLContext_utils import split_X_y_z
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from .TrainInterationInterface import TrainPreperationInterface
from.interfaces.VarianceFilter import VarianceFilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

class MLContext:
    def __init__(self):
        sqlContext, self.train, self.test = load_context()
        self.label_names = ["Risk Level", "Class"] #TODO: AUS DB LADEN
        self.Base = sqlContext["Base"]
        self.metadata = sqlContext["metadata"]
        self.engine = sqlContext["engine"]
        self.session = Session(bind=self.engine)
        sqlContext["session"] = self.session
        self.context = sqlContext
        self.hooks = list()
        self.train_methods = list()
        self.xai_hooks = list()
        self.train_X, self.train_y, self.train_db_indexes = split_X_y_z(self.train)
        self.test_X, self.test_y, self.test_db_indexes = split_X_y_z(self.test)
    
    def start_train_process(self):
        try:
            load_init_parameters(self)
            # Get table references
            train_process_table = self.Base.classes["train_process"]
            train_process_init_parameter_table = self.Base.classes["train_process_init_parameter"]
            # Load initial parameters - use latest parameters if none were explicitly given
            count_process = self.session.query(train_process_table.id).count()
            count_paras = self.session.query(train_process_init_parameter_table.id).count()

            # Fetch the latest init_parameters safely
            self.init_parameters = self.session.query(train_process_init_parameter_table).order_by(train_process_init_parameter_table.id.desc()).first()

            # Clone and add init_parameters if counts are equal
            if count_process == count_paras and self.init_parameters is not None:
                init_parameters_clone = train_process_init_parameter_table()
                for column in train_process_init_parameter_table.__table__.columns:
                    if column.name != "id":
                        setattr(init_parameters_clone, column.name, getattr(self.init_parameters, column.name))
                self.session.add(init_parameters_clone)

            # Create and add new train_process entry
            self.train_process = create_object(self.context, "train_process", with_commit=True)

            # Create and add new train_process_score entry
            # Assuming a one-to-one relationship with train_process
            # train_process_score = create_object(context, "train_process_score", train_process_id=train_process.id)

            # Committing the changes
            self.session.commit()

            # Update initial parameters if available
            if self.init_parameters:
                update_object_attributes(self.context, self.init_parameters, 
                                         train_process_id = self.train_process.id)
            
            #SetupTrainProcess - Create Training Sample
            used_indexes = sample_snapshot(self)
            for i, index in enumerate(used_indexes):            
                train_process_train_data_junction = create_object(self.context, "datapoint_train_process_junction",
                                                                train_process_id = self.train_process.id,
                                                                train_data_id = index + 1)
                self.session.commit()
            
            self.iter_args = dict()
            self.iter_objs = dict()
            self.iter_train_X = dict()
            self.iter_train_y = dict()
            self.iter_test_X = dict()
            self.iter_test_y = dict()
            
            for i in range(3): 
                self.iter_args[i] = dict()
                self.iter_objs[i] = dict()    
                self.iter_objs[i]["model"] = dict()               
                self.start_train_iteration(i)
                
        except SQLAlchemyError as e:
            self.session.rollback()  # Rollback in case of error
            print(f"An error occurred: {e}")
        finally:
            self.session.close()  # Ensure session is closed after the operation)
            
    def start_train_iteration(self, i):
        #This runs multiple times per Object!        
        #CreateSQLObjects create_train_iteration_objects(mlContext)
        create_train_iteration_objects(self, i)
        self.session.commit()
        
        #CreateTrainingObjects
        
        self.iter_train_X[i] = self.train_X
        self.iter_train_y[i] = self.train_y
        self.iter_test_X[i] = self.test_X
        self.iter_test_y[i] = self.test_y                 
        
        for hook in self.hooks:
            hook.populate(i)
        for train_method in self.train_methods:
            train_method.populate(i)
        
        self.iter_args[i]["mlContext"] = self
        self.iter_args[i]["i"] = i   
        
        for train_method in self.train_methods:
            trials = Trials()
            tm_dic = dict()
            tm_dic["train_method"] = train_method
            args = {**self.iter_args[i], **tm_dic}
            params = fmin(callback, args, algo = tpe.suggest, max_evals = 2, trials = trials)
            train_method.upload(i)
        
        

def callback(args):   
    #IMPORTANT: iterargs != args; args contains the selected parameters by hyperopt iterargs contains the definitions; use args for referening these otherwise iterargs;
    #TODO: Refactor dictionary structure to typed objects
    i = args["i"]
    mlContext = args["mlContext"]
    train_method = args["train_method"]
    
    
    for hook in args["mlContext"].hooks:
        hook.calculate(i, args)
        
    acc = 1 - train_method.calculate(i, args)
    return {'loss': acc, 'status': STATUS_OK}
