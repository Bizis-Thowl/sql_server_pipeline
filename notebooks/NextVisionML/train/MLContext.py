from .load_context import load_context
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from ..util import create_object, get_next_ID_for_Table, update_object_attributes
from ..util import update_object_attributes
from .MLContext_sql_utils import create_train_iteration_objects, load_init_parameters
from .train_utils import sample_snapshot
from .MLContext_utils import split_X_and_y
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from .TrainInterationInterface import TrainInterationInterface
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
        self.xai_hooks = list()
        self.train_X, self.train_y = split_X_and_y(self.train)
        self.test_X, self.test_y = split_X_and_y(self.test)
    
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

        self.iter_args[i] =  {
            "max_depth": hp.choice('max_depth', range(1,100)),
            "min_samples_leaf": hp.choice("min_samples_leaf", range(1,15)),
            "random_state": hp.randint("random_state", 3000),
            "max_features": hp.choice('max_features', range(1,50)),
            "criterion": hp.choice('criterion', ["gini", "entropy"]),
            #"variance_threshold_var_fac": hp.randint("variance_threshold", 100)
        }         
        
        for hook in self.hooks:
            hook.populate(i)
                
        
        self.iter_args[i]["mlContext"] = self
        self.iter_args[i]["i"] = i 
        
        trials = Trials()
        params = fmin(callback, dict(self.iter_args[i]), algo = tpe.suggest, max_evals = 10, trials = trials)
        
        for hook in self.xai_hooks:
            hook.call()
            
        

def callback(args):   
    i = args["i"]
    mlContext = args["mlContext"]
    
    for hook in args["mlContext"].hooks:
        hook.calculate(i, args)
        
    update_object_attributes(mlContext.context, mlContext.iter_objs[i]["hyperparameter"], 
                    max_depth = int(args["max_depth"]),
                    min_samples_leaf = int(args["min_samples_leaf"]),
                    random_state = int(args["random_state"]),
                    max_features = int(args["max_features"]),
                    criterion = args["criterion"],    
    )
    
    dtr = DecisionTreeClassifier(
        max_depth = args["max_depth"],
        min_samples_leaf = args["min_samples_leaf"],
        random_state = args["random_state"],
        max_features = args["max_features"],
        criterion = args["criterion"],
    )
    dtr.fit(mlContext.iter_train_X[i], mlContext.iter_train_y[i])
    mlContext.iter_objs[i]["model"] = dtr
    eval_predict = dtr.predict(mlContext.iter_test_X[i]) 
       
    #TODO:Evaluate using full train data vs validation data or split test data...; intention:validation integrity, validation data is incorporated in trainings process
    accuracy = balanced_accuracy_score(mlContext.iter_test_y[i], eval_predict) 
    
    #args["train_process_iteration_score"].balanced_accuracy_score = accuracy TODO: an model_scores anhägen
    
    update_object_attributes(mlContext.context, mlContext.iter_objs[i]["train_process_iteration_score"],
                             balanced_accuracy_score = accuracy)
    #session.commit()
    acc = 1-accuracy
    return {'loss': acc, 'status': STATUS_OK}