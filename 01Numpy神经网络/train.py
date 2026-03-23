# 1.DataLoader读入并处理数据 -> DataLoader
# 2.数据进入网络，前向传播一次，得到y_hat，与y计算loss -> model,loss
# 3.反向传播，计算梯度 -> optimizer
# 4.优化器，梯度下降优化参数 -> optimizer
# 5.epoch->epochs，有batch，可视化，TB -> train
# 6.参数管理 -> utils
from DataLoader import *
from model import *
from loss import *
from utils import *

def train_one_epoch(model, loss, X_train, y_train, batch_size):
    batch_losses = []

    # ===== 手写 mini-batch 梯度下降 =====
    for X_batch, y_batch in iterate_batch(X_train, y_train, batch_size=batch_size, shuffle=True):
        y_pred, _ = model.forward(X_batch, training=True)

        loss_batch = loss(y_true=y_batch, y_pred=y_pred, model=model)

        model.backward(y_true=y_batch, y_pred=y_pred)

        # ===== L2 正则化梯度统一在 loss.py 中添加 =====
        loss.apply_l2_grad(model)

        model.update()
        batch_losses.append(loss_batch)

    return float(np.mean(batch_losses))

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

    return loss_epoch, acc_epoch, y_true_all, y_pred_all

def train_epochs(args, model, loss, X_train, y_train, X_test, y_test, writer=None):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = -1.0
    best_y_true = None
    best_y_pred = None

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            loss=loss,
            X_train=X_train,
            y_train=y_train,
            batch_size=args.batch_size
        )

        val_loss, val_acc, y_true_all, y_pred_all = evaluate_one_epoch(
            model=model,
            loss=loss,
            X_eval=X_test,
            y_eval=y_test,
            batch_size=args.batch_size
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Val", val_loss, epoch + 1)
            writer.add_scalar("Accuracy/Val", val_acc, epoch + 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_y_true = y_true_all.copy()
            best_y_pred = y_pred_all.copy()

        print(
            f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}  "
            f"val_acc={val_acc:.4f}"
        )

    return history, best_y_true, best_y_pred


if __name__ == '__main__':
    # 1.
    args = build_args()
    save_args_to_json(args)

    make_dir(args.save_dir)
    make_dir(args.log_dir)

    Dataloader = sklearnDataLoader(data_home=args.data_home)
    X_train, y_train, X_test, y_test = Dataloader.Load()

    # train epochs
        # train one epoch
    # 2. 超参数创建和网络创建

    activation_fn, d_activation_fn = get_activation_functions(args.activation)

    model = Net(
        hidden_num1=args.hidden_num1,
        hidden_num2=args.hidden_num2,
        activation=activation_fn,
        d_activation=d_activation_fn,
        lr=args.lr,
        method=args.method,
        dropout_rate=args.dropout_rate
    )
    # 3. 训练
    loss = CrossEntropyLoss(l2_lambda=args.l2_lambda)

    writer = create_summary_writer(args.log_dir)

    history, best_y_true, best_y_pred = train_epochs(
        args=args,
        model=model,
        loss=loss,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        writer=writer
    )

    plot_training_history(
        history=history,
        save_path=f"{args.save_dir}/training_curves.png"
    )

    save_confusion_matrix(
        y_true=best_y_true,
        y_pred=best_y_pred,
        save_path=f"{args.save_dir}/confusion_matrix.png",
        class_names=[str(i) for i in range(10)]
    )

    save_misclassified_examples(
        X=X_test,
        y_true=best_y_true,
        y_pred=best_y_pred,
        save_path=f"{args.save_dir}/misclassified_examples.png",
        max_show=25
    )

    close_summary_writer(writer)