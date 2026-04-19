from DataLoader import *
from loss import *
from model import *
from utils import *


def train_one_epoch(model, loss, X_train, y_train, batch_size):
    batch_losses = []
    all_y_pred = []
    all_y_true = []

    for X_batch, y_batch in iterate_batch(X_train, y_train, batch_size=batch_size, shuffle=True):
        y_pred, _ = model.forward(X_batch, training=True)
        loss_batch = loss(y_true=y_batch, y_pred=y_pred, model=model)

        model.backward(y_true=y_batch, y_pred=y_pred)
        loss.apply_l2_grad(model)
        model.update()

        batch_losses.append(loss_batch)
        all_y_pred.append(y_pred)
        all_y_true.append(y_batch)

    y_pred_all = np.concatenate(all_y_pred, axis=0)
    y_true_all = np.concatenate(all_y_true, axis=0)
    loss_epoch = float(np.mean(batch_losses))
    acc_epoch = accuracy(y_pred_all, y_true_all)

    return loss_epoch, acc_epoch


def evaluate_one_epoch(model, loss, X_eval, y_eval, batch_size):
    batch_losses = []
    all_y_pred = []
    all_y_true = []

    for X_batch, y_batch in iterate_batch(X_eval, y_eval, batch_size=batch_size, shuffle=False):
        y_pred, _ = model.forward(X_batch, training=False)
        loss_batch = loss(y_true=y_batch, y_pred=y_pred, model=model)

        batch_losses.append(loss_batch)
        all_y_pred.append(y_pred)
        all_y_true.append(y_batch)

    y_pred_all = np.concatenate(all_y_pred, axis=0)
    y_true_all = np.concatenate(all_y_true, axis=0)

    loss_epoch = float(np.mean(batch_losses))
    acc_epoch = accuracy(y_pred_all, y_true_all)

    sample_count = min(10, X_eval.shape[0])
    sample_X = X_eval[:sample_count].copy()
    sample_y_true = y_true_all[:sample_count].copy()
    sample_y_pred = y_pred_all[:sample_count].copy()

    return loss_epoch, acc_epoch, y_true_all, y_pred_all, sample_X, sample_y_true, sample_y_pred


def train_epochs(args, model, loss, X_train, y_train, X_test, y_test, run=None):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_epoch = 0
    best_y_true = None
    best_y_pred = None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loss=loss,
            X_train=X_train,
            y_train=y_train,
            batch_size=args.batch_size,
        )

        val_loss, val_acc, y_true_all, y_pred_all, sample_X, sample_y_true, sample_y_pred = evaluate_one_epoch(
            model=model,
            loss=loss,
            X_eval=X_test,
            y_eval=y_test,
            batch_size=args.batch_size,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_y_true = y_true_all.copy()
            best_y_pred = y_pred_all.copy()

        if run is not None:
            run.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                    "val_samples": build_wandb_image_payload(
                        X=sample_X,
                        y_true=sample_y_true,
                        y_pred=sample_y_pred,
                        max_show=10,
                    ),
                },
                step=epoch + 1,
            )

        print(
            f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f}  "
            f"train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.6f}  "
            f"val_acc={val_acc:.4f}  "
            f"best_val_acc={best_val_acc:.4f}"
        )

    summary = {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "final_val_acc": float(history["val_acc"][-1]),
        "final_train_acc": float(history["train_acc"][-1]),
    }

    return history, best_y_true, best_y_pred, summary


def run_training(args):
    args = finalize_run_paths(args)
    make_dir(args.save_dir)
    make_dir(args.log_dir)
    save_args_to_json(args)

    dataloader = sklearnDataLoader(data_home=args.data_home, seed=args.seed)
    X_train, y_train, X_test, y_test = dataloader.Load()

    set_random_seed(args.seed)
    activation_fn, d_activation_fn = get_activation_functions(args.activation)
    model = Net(
        hidden_num1=args.hidden_num1,
        hidden_num2=args.hidden_num2,
        activation=activation_fn,
        d_activation=d_activation_fn,
        lr=args.lr,
        method=args.method,
        dropout_rate=args.dropout_rate,
    )
    loss = CrossEntropyLoss(l2_lambda=args.l2_lambda)

    run = init_wandb_run(args)

    try:
        history, best_y_true, best_y_pred, summary = train_epochs(
            args=args,
            model=model,
            loss=loss,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            run=run,
        )

        training_curve_path = f"{args.save_dir}/training_curves.png"
        confusion_matrix_path = f"{args.save_dir}/confusion_matrix.png"
        misclassified_examples_path = f"{args.save_dir}/misclassified_examples.png"

        plot_training_history(history=history, save_path=training_curve_path)
        save_confusion_matrix(
            y_true=best_y_true,
            y_pred=best_y_pred,
            save_path=confusion_matrix_path,
            class_names=[str(i) for i in range(10)],
        )
        save_misclassified_examples(
            X=X_test,
            y_true=best_y_true,
            y_pred=best_y_pred,
            save_path=misclassified_examples_path,
            max_show=25,
        )

        if run is not None:
            run.summary.update(summary)
            run.log(
                {
                    "training_curves": wandb.Image(training_curve_path),
                    "confusion_matrix": wandb.Image(confusion_matrix_path),
                },
                step=args.epochs,
            )

        return {
            "history": history,
            "summary": summary,
            "paths": {
                "training_curves": training_curve_path,
                "confusion_matrix": confusion_matrix_path,
                "misclassified_examples": misclassified_examples_path,
            },
        }
    finally:
        finish_wandb_run(run)


if __name__ == "__main__":
    run_training(build_args())
