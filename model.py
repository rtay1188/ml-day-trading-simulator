import logging
import sys
import warnings

import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import NHiTS, TimeSeriesDataSet
from pytorch_forecasting.metrics import RMSE, MAE, SMAPE

import config
import db

def add_tidx_and_groupid(df:pd.DataFrame)->pd.DataFrame:
    """
    Adds 'group_id' and 'time_idx' columns to the input DataFrame based on the 'timestamp' column.

    'group_id' is assigned per unique date, and 'time_idx' is a time index within each group.
    The original 'timestamp' column is dropped.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'timestamp' column.

    Returns:
        pd.DataFrame: Modified DataFrame with 'group_id' and 'time_idx' columns.
    """
    new_df=df.copy()
    unique_dates=new_df['timestamp'].dt.date.unique()
    date_to_group = {date: i for i, date in enumerate(sorted(unique_dates))}
    new_df['group_id'] = new_df['timestamp'].dt.date.map(date_to_group)
    new_df['time_idx'] = new_df.groupby('group_id').cumcount()
    new_df = new_df.drop(columns=['timestamp'])
    return new_df

def split_columns_by_group_variation(df:pd.DataFrame, group_col='group_id')->tuple[list[str], list[str]]:
    """
    Splits DataFrame columns into time-varying and static based on whether values vary within each group.

    Args:
        df (pd.DataFrame): Input DataFrame.
        group_col (str): Column name that defines the group.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - time_varying: columns that vary within groups
            - static: columns that remain constant within each group
    """
    time_varying = []
    static = []
    for col in df.columns:
        if col == group_col:
            continue
        unique_counts = df.groupby(group_col)[col].nunique()
        if (unique_counts <= 1).all():
            static.append(col)
        else:
            time_varying.append(col)
    return time_varying, static

def construct_dataframes(data: pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input DataFrame into training and validation sets based on 'group_id'.

    Groups with 'group_id' < 7 are used for training. Validation groups are remapped to start at 0.
    'group_id' is also converted to string type in both sets.

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'group_id' column.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (train_df, val_df).
    """
    train_df = data[data['group_id']<7].copy()
    val_df = data[data['group_id']>=7].copy()
    val_df['group_id'] = val_df['group_id'] - val_df['group_id'].min()
    train_df['group_id'] = train_df['group_id'].astype(str)
    val_df['group_id'] = val_df['group_id'].astype(str)
    return train_df, val_df

def construct_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, tvur: list[str], sr: list[str], context_length_datapoints: int, prediction_length: int)->tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Constructs PyTorch Forecasting TimeSeriesDataSet objects for training and validation.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        tvur (list[str]): Time-varying unknown real-valued column names.
        sr (list[str]): Static real-valued column names.
        context_length_datapoints (int): Length of input history for each sample.
        prediction_length (int): Number of time steps to predict.

    Returns:
        tuple[TimeSeriesDataSet, TimeSeriesDataSet]: Training and validation datasets.
    """
    pl.seed_everything(42)
    train_dataset=TimeSeriesDataSet(
        data=train_df,
        time_idx="time_idx",
        group_ids=["group_id"],
        target="currentPrice",
        max_encoder_length=context_length_datapoints,
        max_prediction_length=prediction_length,
        static_reals=sr,
        static_categoricals=["group_id"],
        time_varying_unknown_reals=tvur,
        predict_mode=False
    )
    val_dataset=TimeSeriesDataSet(
        data=val_df,
        time_idx="time_idx",
        group_ids=["group_id"],
        target="currentPrice",
        max_encoder_length=context_length_datapoints,
        max_prediction_length=prediction_length,
        static_reals=sr,
        static_categoricals=["group_id"],
        time_varying_unknown_reals=tvur,
        predict_mode=False
    )
    return(train_dataset, val_dataset)

def construct_model(train_dataset: TimeSeriesDataSet, hidden_size: int)->NHiTS:
    """
    Constructs an N-HiTS model using the provided training dataset.

    Args:
        train_dataset (TimeSeriesDataSet): The dataset to base model configuration on.
        hidden_size (int): Hidden size of the model layers.

    Returns:
        NHiTS: Initialized N-HiTS model ready for training.
    """
    model = NHiTS.from_dataset(
        train_dataset,
        learning_rate=5e-4,
        loss=SMAPE(),
        hidden_size=hidden_size,
        dropout=0.2
    )
    return model

def construct_dataloaders(train_dataset: TimeSeriesDataSet, val_dataset: TimeSeriesDataSet, batch_size: int)->tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Creates PyTorch dataloaders for training and validation datasets.

    Args:
        train_dataset (TimeSeriesDataSet): Training dataset.
        val_dataset (TimeSeriesDataSet): Validation dataset.
        batch_size (int): Batch size for both dataloaders.

    Returns:
        tuple[DataLoader, DataLoader]: Tuple containing training and validation dataloaders.
    """
    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=8, persistent_workers=True)
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=8)
    return train_dataloader, val_dataloader

def fit_model(nhits_model: NHiTS, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader, set_max_epochs: int)->str:
    """
    Fits an N-HiTS model using the provided dataloaders.

    Args:
        nhits_model (NHiTS): The model to be trained.
        train_dataloader (DataLoader): Dataloader for training.
        val_dataloader (DataLoader): Dataloader for validation.
        set_max_epochs (int): Number of epochs to train.

    Returns:
        str: Path to the best model checkpoint.
    """
    trainer=pl.Trainer(
        max_epochs=set_max_epochs,
        accelerator="cpu",
        enable_progress_bar=True,
    )
    trainer.fit(nhits_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    best_model_path = trainer.checkpoint_callback.best_model_path
    return best_model_path

def train_best_model(cfg: config.BaseConfig)->tuple[str, TimeSeriesDataSet, pd.DataFrame]:
    """
    Trains an N-HiTS time series forecasting model using data from a PostgreSQL database.

    This function performs the following steps:
      1. Suppresses PyTorch Lightning logs and common data loader warnings.
      2. Connects to the database and fetches raw stock data.
      3. Preprocesses the data by adding time and group indices.
      4. Splits the data into training and validation sets.
      5. Identifies static and time-varying features by group.
      6. Constructs PyTorch Forecasting datasets and dataloaders.
      7. Initializes and trains the N-HiTS model.
      8. Safely disconnects from the databsse.
      9. Returns the path to the best-performing model checkpoint.

    Args:
        cfg (BaseConfig): Configuration object containing model and data settings.

      
    Returns:
        tuple[str, TimeSeriesDataSet, pd.DataFrame]: Tuple containing the filesystem path to the best model checkpoint file, the training dataset, and validation dataframe.

    """
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    conn, cur = db.connect_to_db()
    raw_data = db.fetch_data(cur, cfg.quote_table)
    processed_data=add_tidx_and_groupid(raw_data)
    train_df, val_df = construct_dataframes(processed_data)
    tvur, sr = split_columns_by_group_variation(processed_data)
    train_dataset, val_dataset=construct_datasets(train_df, val_df, tvur, sr, cfg.context_length, cfg.prediction_length)
    nhits_model=construct_model(train_dataset, hidden_size=512)
    train_dataloader, val_dataloader = construct_dataloaders(train_dataset, val_dataset, batch_size=64)
    best_model_path=fit_model(nhits_model, train_dataloader, val_dataloader, set_max_epochs=10)
    db.disconnect_from_db(conn, cur)
    return best_model_path, train_dataset, val_df

def create_future_rows(last_row: pd.Series, prediction_length: int) -> pd.DataFrame:
    """
    Generate future rows by incrementing the 'time_idx' field of the last known row.

    This function creates a DataFrame of future time steps for prediction purposes
    by duplicating the last known data row and incrementing its 'time_idx' field
    from (last_row["time_idx"] + 1) to (last_row["time_idx"] + prediction_length).

    Args:
        last_row (pd.Series): The last row of data to use as a template for future rows.
        prediction_length (int): The number of future rows to generate.

    Returns:
        pd.DataFrame: A DataFrame containing `prediction_length` future rows with updated time indices.
    """
    future_rows = []
    for i in range(1, prediction_length + 1):
        new_row = last_row.copy()
        new_row["time_idx"] = last_row["time_idx"] + i
        future_rows.append(new_row)

    return pd.DataFrame(future_rows)

def make_prediction(model: NHiTS, context_df: pd.DataFrame, train_dataset: TimeSeriesDataSet, prediction_length: int)->float:
    """
    Generate a single future prediction using the NHiTS model and a given historical context.

    This function appends future time steps to the provided context DataFrame, constructs
    a prediction dataset using the TimeSeriesDataSet, and performs inference with the given model.
    It returns the first predicted value as a float.

    Args:
        model (NHiTS): A trained NHiTS model from PyTorch Forecasting.
        context_df (pd.DataFrame): A DataFrame containing the historical context data.
        train_dataset (TimeSeriesDataSet): The original training dataset, used to ensure consistency in preprocessing.
        prediction_length (int): The number of future time steps to predict.

    Returns:
        float: The first predicted value from the model's output.
    """
    last_row = context_df.iloc[-1].copy()
    pred_rows_df = create_future_rows(last_row, prediction_length)
    input_df = pd.concat([context_df, pred_rows_df], ignore_index=True)
    predict_ds = TimeSeriesDataSet.from_dataset(
        train_dataset,
        input_df,
        stop_randomization=True,
        predict=True
    )
    predict_dl = predict_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    prediction = model.predict(predict_dl, trainer_kwargs={"logger": False, "enable_checkpointing":False, "enable_model_summary":False, "enable_progress_bar":False})
    fl = (prediction[0][0]).item()
    return fl

def rolling_predictions(
    nhits: NHiTS,
    val_df: pd.DataFrame,
    train_dataset: TimeSeriesDataSet,
    group_id: str,
    context_length: int,
    prediction_length: int
) -> list:
    """
    Step through the validation set and make a prediction at each step using context_length.

    Args:
        nhits: Trained NHiTS model.
        val_df: DataFrame with validation data.
        train_dataset: Original training TimeSeriesDataSet for metadata reuse.
        group_id: Which group to run predictions on (as string).
        context_length: Number of time steps used as model input.
        prediction_length: Number of time steps to predict ahead.

    Returns:
        List of (time_idx, prediction) tuples.
    """
    closing_price = 0
    df_group = val_df[val_df["group_id"] == group_id].reset_index(drop=True)
    results = [] # tidx, predicted value at tidx + 1, currentPrice at tidx - predicted value at tidx + 1, accountValue after trade, action from trade

    max_idx = len(df_group) - context_length + 1

    for i in range(max_idx):
        context_df = df_group.iloc[i : i + context_length].copy()
        pred_value = make_prediction(nhits, context_df, train_dataset, prediction_length)
        prediction_time_idx = context_df.iloc[-1]["time_idx"]
        closing_price= context_df.iloc[-1]["currentPrice"]
        results.append((prediction_time_idx, pred_value, closing_price - pred_value))

    return results

def analyze(results, val_df)->float:
    """
    Analyzes the results of prediction on the validation set to get the corrective offset.

    Returns:
        Corrective offset which is the mean of difference between currentPrice at tidx and the predictedPrice for tidx + 1 (float)
    
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=["time_idx", "predicted", "vsCurrent"])
    results_df["time_idx"] = results_df["time_idx"].astype(int)

    # Shift prediction time to match the actual it is predicting
    results_df["predicted_time_idx"] = results_df["time_idx"] + 1

    # Filter val_df for the correct group
    val_group = val_df[val_df["group_id"] == "0"].copy()
    val_group = val_group.drop_duplicates(subset=["time_idx"])

    # Merge using shifted time index
    comparison_df = pd.merge(
        results_df,
        val_group[["time_idx", "currentPrice"]],
        left_on="predicted_time_idx",
        right_on="time_idx",
        how="left",
        suffixes=("_pred", "_actual")
    )

    # Optional comparison
    comparison_df["error"] = comparison_df["predicted"] - comparison_df["currentPrice"]
    comparison_df["above_actual"] = comparison_df["error"] > 0
    comparison_df["below_input"]=comparison_df["vsCurrent"] > 0

    # Compute comparison
    comparison_df["vsNextCurrentPrice"] = comparison_df["error"].apply(
        lambda x: "above" if x > 0 else ("below" if x < 0 else "equal")
    )
    comparison_df["vsCurrentPrice"] = comparison_df["vsCurrent"].apply(
        lambda x: "below" if x > 0 else ("above" if x < 0 else "equal")
    )

    # Get percentage breakdown
    percentages = comparison_df["vsNextCurrentPrice"].value_counts(normalize=True) * 100
    percentages_curr = comparison_df["vsCurrentPrice"].value_counts(normalize=True) * 100

    offset = results_df["vsCurrent"].mean()

    sys.stderr.write(percentages.to_string() + "\n")
    sys.stderr.write(percentages_curr.to_string() + "\n")
    sys.stderr.write(f"{offset}\n")
    return offset