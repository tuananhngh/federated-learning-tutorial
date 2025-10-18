import torch
from collections import OrderedDict
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


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    assert isinstance(msg.content["arrays"], ArrayRecord)
    assert isinstance(msg.content["config"], ConfigRecord)
    assert isinstance(msg.content["metrics"], MetricRecord)

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
        trainloader, _ = load_data(partition_id, num_partitions, batch_size)
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
    assert isinstance(msg.content["config"], ConfigRecord)
    assert isinstance(msg.content["metrics"], MetricRecord)

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

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
