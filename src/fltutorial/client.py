import torch
import json
import os
from collections import OrderedDict
from pathlib import Path
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.common.typing import UserConfig
from flwr.clientapp import ClientApp
from fltutorial.task import Net, load_data
from fltutorial.task import test as test_fn
from fltutorial.task import train as train_fn

app = ClientApp()

# Create directory for metrics
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    assert isinstance(msg.content["arrays"], ArrayRecord)

    if "config" in msg.content:
        assert isinstance(msg.content["config"], ConfigRecord)
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    if (
        isinstance(partition_id, int)
        and isinstance(num_partitions, int)
        and isinstance(batch_size, int)
    ):
        print(f"Loading data for partition {partition_id}/{num_partitions}")
        print(f"Batch size: {batch_size}")
        trainloader, _ = load_data(partition_id, num_partitions, batch_size)
        print(f"Loaded {len(trainloader.dataset)} training samples")
    else:
        raise ValueError(
            "partition_id, num_partitions, and batch_size must be integers"
        )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Log metrics for this client
    partition_id = context.node_config["partition-id"]
    server_round = context.state.current_round()

    # Save training metrics
    metrics_file = METRICS_DIR / f"client_{partition_id}_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            client_metrics = json.load(f)
    else:
        client_metrics = {"train_loss": [], "eval_loss": [], "eval_acc": [], "rounds": []}

    client_metrics["train_loss"].append(train_loss)
    client_metrics["rounds"].append(server_round)

    with open(metrics_file, "w") as f:
        json.dump(client_metrics, f, indent=2)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())  # type: ignore
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    assert isinstance(msg.content["arrays"], ArrayRecord)
    # Config and metrics may not always be present in evaluation messages

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    if (
        isinstance(partition_id, int)
        and isinstance(num_partitions, int)
        and isinstance(batch_size, int)
    ):
        _, valloader = load_data(partition_id, num_partitions, batch_size)
    else:
        raise ValueError(
            "partition_id, num_partitions, and batch_size must be integers"
        )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Log evaluation metrics for this client
    partition_id = context.node_config["partition-id"]
    server_round = context.state.current_round()

    # Save evaluation metrics
    metrics_file = METRICS_DIR / f"client_{partition_id}_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            client_metrics = json.load(f)
    else:
        client_metrics = {"train_loss": [], "eval_loss": [], "eval_acc": [], "rounds": []}

    client_metrics["eval_loss"].append(eval_loss)
    client_metrics["eval_acc"].append(eval_acc)

    with open(metrics_file, "w") as f:
        json.dump(client_metrics, f, indent=2)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
