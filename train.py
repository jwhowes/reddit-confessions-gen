import click

from src.trainer import Trainer


@click.command()
@click.argument("config_dir", type=click.Path(exists=True, file_okay=False))
def train(config_dir: str):
    trainer = Trainer(config_dir)

    trainer.train()


if __name__ == "__main__":
    train()
