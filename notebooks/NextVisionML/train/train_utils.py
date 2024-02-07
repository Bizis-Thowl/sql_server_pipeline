def sample_snapshot(mlContext):
    value_counts = mlContext.train_y["Risk Level"].value_counts()
    for i in mlContext.train_y["Risk Level"].unique():
        drop_amount = value_counts[i]-value_counts.min()
        class_df = mlContext.train_y[mlContext.train_y["Risk Level"] == i]
        drop_indexes = class_df.sample(n=drop_amount).index
        mlContext.train_y = mlContext.train_y.drop(drop_indexes)
        mlContext.train_X =  mlContext.train_X.drop(drop_indexes)
    used_indexes = mlContext.train_X.index
    mlContext.train_y = mlContext.train_y.reset_index(drop=True)
    mlContext.train_X = mlContext.train_X.reset_index(drop=True)
    return used_indexes

