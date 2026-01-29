import torch
import torch.optim as optim
from model import *
import numpy as np
import utils
from Params import args
from DataHandler import DataHandler
from sklearn.metrics import f1_score, average_precision_score, accuracy_score

class trainer():
    def __init__(self, device):
        self.handler = DataHandler()
        self.model = STHSL()
        self.model.to(device)
        # Bhavani: Enable debug shapes
        self.model.debug_shapes = True
        print("DEBUG SHAPES status:", self.model.debug_shapes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss = utils.cal_loss_r
        self.metrics = utils.cal_metrics_r

    def sampleTrainBatch(self, batIds, st, ed):
        batch = ed - st
        idx = batIds[0: batch]
        label = self.handler.trnT[:, idx, :]
        label = np.transpose(label, [1, 0, 2])
        retLabels = (label >= 0) * 1
        mask = retLabels
        retLabels = label

        feat_list = []
        for i in range(batch):
            feat_one = self.handler.trnT[:, idx[i] - args.temporalRange: idx[i], :]
            feat_one = np.expand_dims(feat_one, axis=0)
            feat_list.append(feat_one)
        feat_batch = np.concatenate(feat_list, axis=0)
        return self.handler.zScore(feat_batch), retLabels, mask

    def sampTestBatch(self, batIds, st, ed, tstTensor, inpTensor):
        batch = ed - st
        idx = batIds[0: batch]
        label = tstTensor[:, idx, :]
        label = np.transpose(label, [1, 0, 2])
        retLabels = label
        mask = 1 * (label > 0)

        feat_list = []
        for i in range(batch):
            if idx[i] - args.temporalRange < 0:
                temT = inpTensor[:, idx[i] - args.temporalRange:, :]
                temT2 = tstTensor[:, :idx[i], :]
                feat_one = np.concatenate([temT, temT2], axis=1)
            else:
                feat_one = tstTensor[:, idx[i] - args.temporalRange: idx[i], :]
            feat_one = np.expand_dims(feat_one, axis=0)
            feat_list.append(feat_one)
        feats = np.concatenate(feat_list, axis=0)
        return self.handler.zScore(feats), retLabels, mask


    def train(self):
        self.model.train()
        ids = np.random.permutation(list(range(args.temporalRange, args.trnDays)))
        epochLoss, epochPreLoss, epochAcc = [0] * 3
        num = len(ids)
        steps = int(np.ceil(num / args.batch))
        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = ids[st: ed]
            bt = ed - st
            if args.use_ode_option in ("baseline", "option2"):
                Infomax_L1 = torch.ones(bt, args.offNum, args.areaNum)
                Infomax_L2 = torch.zeros(bt, args.offNum, args.areaNum)
                Infomax_labels = torch.Tensor(torch.cat((Infomax_L1, Infomax_L2), -1)).to(args.device)

            tem = self.sampleTrainBatch(batIds, st, ed)
            feats, labels, mask = tem
            mask = torch.Tensor(mask).to(args.device)
            self.optimizer.zero_grad()

            idx = np.random.permutation(args.areaNum)
            DGI_feats = torch.Tensor(feats[:, idx, :, :]).to(args.device)
            feats = torch.Tensor(feats).to(args.device)
            labels = torch.Tensor(labels).to(args.device)

            if args.use_ode_option == "baseline":
                out_local, eb_local, eb_global, Infomax_pred, out_global = self.model(feats, DGI_feats)
            elif args.use_ode_option == "option1":
                out_local, eb_tra_local, eb_tra_global, Infomax_pred, out_global, Z_t = self.model(feats, DGI_feats)
            elif args.use_ode_option == "option2":
                out_local, eb_local, eb_global, Infomax_pred, out_global, fe, Z_t, pred = self.model(feats, DGI_feats)

            # ---- Bhavani: DEBUG SHAPE GUARD ----
            dbg = self.model.debug_shapes and (not hasattr(self, "_dbg_shape_printed"))
            if dbg:
                if args.use_ode_option != "baseline":
                    print("Decoder input Z_t shape:", Z_t.shape)
                    print("Labels shape:", labels.shape)
                    if args.use_ode_option == "option2":
                        print("Decoder output pred shape:", pred.shape)
                    self._dbg_shape_printed = True  # mark printed immediately
            # -------------------------------------
            out_local = self.handler.zInverse(out_local)
            out_global = self.handler.zInverse(out_global)

            if args.use_ode_option == "baseline":
                loss = (utils.Informax_loss(Infomax_pred, Infomax_labels) * args.ir) + (utils.infoNCEloss(eb_global, eb_local) * args.cr) + \
                self.loss(out_local, labels, mask) + self.loss(out_global, labels, mask)
            elif args.use_ode_option == "option1":
                loss = (utils.Informax_loss_option1(Infomax_pred) * args.ir) + (utils.infoNCEloss(eb_tra_global, eb_tra_local) * args.cr) + \
                self.loss(out_local, labels, mask) + self.loss(out_global, labels, mask)
            elif args.use_ode_option == "option2":
                loss = (utils.Informax_loss(Infomax_pred, Infomax_labels) * args.ir) + (utils.infoNCEloss(eb_global, eb_local) * args.cr) + \
                self.loss(out_local, labels, mask) + self.loss(pred, labels, mask)
            else:
                raise ValueError("Unknown ODE option {args.use_ode_option}")

            loss.backward()
            self.optimizer.step()
            print('Step %d/%d: preLoss = %.4f         ' % (i, steps, loss), end='\r')
            epochLoss += loss
        epochLoss = epochLoss / steps
        return epochLoss, loss.item()


    def eval(self, iseval, isSparsity):
        self.model.eval()
        want_error = args.eval_metrics in ("error", "all")
        want_acc   = args.eval_metrics in ("accuracy", "all")
        if args.eval_metrics == "none":
            want_error = False
            want_acc = False

        if iseval:
            ids = np.array(list(range(self.handler.valT.shape[1])))
        else:
            ids = np.array(list(range(self.handler.tstT.shape[1])))
        epochLoss, epochPreLoss, = [0] * 2

        num = len(ids)
        if isSparsity:
            epochSqLoss1, epochAbsLoss1, epochTstNum1, epochApeLoss1, epochPosNums1 = [np.zeros(4) for i in range(5)]
            epochSqLoss2, epochAbsLoss2, epochTstNum2, epochApeLoss2, epochPosNums2 = [np.zeros(4) for i in range(5)]
            epochSqLoss3, epochAbsLoss3, epochTstNum3, epochApeLoss3, epochPosNums3 = [np.zeros(4) for i in range(5)]
            epochSqLoss4, epochAbsLoss4, epochTstNum4, epochApeLoss4, epochPosNums4 = [np.zeros(4) for i in range(5)]
            epochSqLoss, epochAbsLoss, epochTstNum, epochApeLoss, epochPosNums = [np.zeros(4) for i in range(5)]
        else:
            epochSqLoss, epochAbsLoss, epochTstNum, epochApeLoss, epochPosNums = [np.zeros(4) for i in range(5)]

        steps = int(np.ceil(num / args.batch))

        n_batches = 0

        # ================= ACC METRIC CONTAINERS =================
        if want_acc:
            all_labels = []
            all_preds  = []
            all_scores = []

            C = args.offNum

            # per-category
            per_cat_labels = {c: [] for c in range(C)}
            per_cat_preds  = {c: [] for c in range(C)}
            per_cat_scores = {c: [] for c in range(C)}

            if isSparsity:
                # per-sparsity
                per_mask_labels = {k: [] for k in (1, 2, 3, 4)}
                per_mask_preds  = {k: [] for k in (1, 2, 3, 4)}
                per_mask_scores = {k: [] for k in (1, 2, 3, 4)}

                # category × sparsity (4×4)
                per_cat_mask_labels = {(c, k): [] for c in range(C) for k in (1, 2, 3, 4)}
                per_cat_mask_preds  = {(c, k): [] for c in range(C) for k in (1, 2, 3, 4)}
                per_cat_mask_scores = {(c, k): [] for c in range(C) for k in (1, 2, 3, 4)}
                
        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = ids[st: ed]

            if iseval:
                tem = self.sampTestBatch(batIds, st, ed, self.handler.valT, self.handler.trnT)
            else:
                tem = self.sampTestBatch(batIds, st, ed, self.handler.tstT, np.concatenate([self.handler.trnT, self.handler.valT], axis=1))
            feats, labels, mask = tem
            idx = np.random.permutation(args.areaNum)
            shuf_feats = feats[:, idx, :, :]
            feats = torch.Tensor(feats).to(args.device)
            shuf_feats = torch.Tensor(shuf_feats).to(args.device)
            if args.use_ode_option == "baseline":
                out_local, eb_local, eb_global, DGI_pred, out_global = self.model(feats, shuf_feats)
            elif args.use_ode_option == "option1":
                out_local, eb_local, eb_global, DGI_pred, out_global, Z_t = self.model(feats, shuf_feats)
            elif args.use_ode_option == "option2":
                out_local, eb_local, eb_global, DGI_pred, out_global, fe , Z_t, pred = self.model(feats, shuf_feats)
            else:
                raise ValueError("Unknown ODE option {args.use_ode_option}")

            if args.use_ode_option in ("baseline", "option1"):
                output = self.handler.zInverse(out_global)
            elif args.use_ode_option == "option2":
                output = self.handler.zInverse(pred)
            else:
                raise ValueError("Unknown ODE option {args.use_ode_option}")
            
            # --- collect overall accuracy metrics ---
            if want_acc:
                lbl_bin  = (labels > args.f1_label_threshold).astype(int)
                pred_bin = (output.cpu().detach().numpy() > args.f1_pred_threshold).astype(int)
                scores   = output.cpu().detach().numpy()

                # overall
                all_labels.append(lbl_bin.reshape(-1))
                all_preds.append(pred_bin.reshape(-1))
                all_scores.append(scores.reshape(-1))

                # per-category
                for c in range(C):
                    per_cat_labels[c].append(lbl_bin[:, :, c].reshape(-1))
                    per_cat_preds[c].append(pred_bin[:, :, c].reshape(-1))
                    per_cat_scores[c].append(scores[:, :, c].reshape(-1))

                # per-sparsity + category×sparsity
                if isSparsity:
                    masks = {
                        1: self.handler.mask1.astype(bool),
                        2: self.handler.mask2.astype(bool),
                        3: self.handler.mask3.astype(bool),
                        4: self.handler.mask4.astype(bool),
                    }

                    for k, mk in masks.items():
                        for c in range(C):
                            a_mask = mk[:, c]
                            # category × mask (for 4×4 table)
                            per_cat_mask_labels[(c, k)].append(lbl_bin[:, :, c][:, a_mask].reshape(-1))
                            per_cat_mask_preds[(c, k)].append(pred_bin[:, :, c][:, a_mask].reshape(-1))
                            per_cat_mask_scores[(c, k)].append(scores[:, :, c][:, a_mask].reshape(-1))
                            per_mask_labels[k].append(lbl_bin[:, :, c][:, a_mask].reshape(-1))
                            per_mask_preds[k].append(pred_bin[:, :, c][:, a_mask].reshape(-1))
                            per_mask_scores[k].append(scores[:, :, c][:, a_mask].reshape(-1))
                            # [Bhavani] try this option instead of above 6 lines when GPU execution to see if there's difference
                            # a_mask = mk[:, c].astype(bool)     # shape: (A,)

                            # # Select exactly the regions belonging to this sparsity-bin for this category
                            # y_sel = lbl_bin[:, a_mask, c]      # shape: (B, Rk)
                            # p_sel = pred_bin[:, a_mask, c]
                            # s_sel = scores[:, a_mask, c]

                            # # Per-mask aggregates (across categories too, by appending category-wise)
                            # per_mask_labels[k].append(y_sel.reshape(-1))
                            # per_mask_preds[k].append(p_sel.reshape(-1))
                            # per_mask_scores[k].append(s_sel.reshape(-1))

                            # # Category × Mask aggregates
                            # per_cat_mask_labels[(c, k)].append(y_sel.reshape(-1))
                            # per_cat_mask_preds[(c, k)].append(p_sel.reshape(-1))
                            # per_cat_mask_scores[(c, k)].append(s_sel.reshape(-1))

            if want_error:
                n_batches += 1
                loss, sqLoss, absLoss, tstNums, apeLoss, posNums = self.metrics(output.cpu().detach().numpy(), labels, mask)
                epochSqLoss += sqLoss
                epochAbsLoss += absLoss
                epochTstNum += tstNums
                epochApeLoss += apeLoss
                epochPosNums += posNums
                epochLoss += loss
                print('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, loss), end='\r')

                if isSparsity:
                    # mask-wise error metrics only when sparsity is enabled
                    _, sqLoss1, absLoss1, tstNums1, apeLoss1, posNums1 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, self.handler.mask1)
                    _, sqLoss2, absLoss2, tstNums2, apeLoss2, posNums2 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, self.handler.mask2)
                    _, sqLoss3, absLoss3, tstNums3, apeLoss3, posNums3 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, self.handler.mask3)
                    _, sqLoss4, absLoss4, tstNums4, apeLoss4, posNums4 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, self.handler.mask4)

                    epochSqLoss1 += sqLoss1
                    epochAbsLoss1 += absLoss1
                    epochTstNum1 += tstNums1
                    epochApeLoss1 += apeLoss1
                    epochPosNums1 += posNums1

                    epochSqLoss2 += sqLoss2
                    epochAbsLoss2 += absLoss2
                    epochTstNum2 += tstNums2
                    epochApeLoss2 += apeLoss2
                    epochPosNums2 += posNums2

                    epochSqLoss3 += sqLoss3
                    epochAbsLoss3 += absLoss3
                    epochTstNum3 += tstNums3
                    epochApeLoss3 += apeLoss3
                    epochPosNums3 += posNums3

                    epochSqLoss4 += sqLoss4
                    epochAbsLoss4 += absLoss4
                    epochTstNum4 += tstNums4
                    epochApeLoss4 += apeLoss4
                    epochPosNums4 += posNums4
        if want_error and n_batches > 0:
            epochLoss = epochLoss / n_batches

        ret = dict()

        if want_acc:
            labels_flat = np.concatenate(all_labels)
            preds_flat  = np.concatenate(all_preds)
            scores_flat = np.concatenate(all_scores)

            ret["MacroF1"] = f1_score(labels_flat, preds_flat, average="macro")
            ret["MicroF1"] = f1_score(labels_flat, preds_flat, average="micro")
            ret["AP"] = average_precision_score(labels_flat, scores_flat)
            ret["Acc"] = accuracy_score(labels_flat, preds_flat)

            # ---- per-category ----
            for c in range(C):
                y = np.concatenate(per_cat_labels[c])
                p = np.concatenate(per_cat_preds[c])
                s = np.concatenate(per_cat_scores[c])

                ret[f"F1_cate_{c}"] = f1_score(y, p, zero_division=0)
                ret[f"Acc_cate_{c}"] = accuracy_score(y, p) if y.size > 0 else float("nan")
                ret[f"AP_cate_{c}"] = (
                    average_precision_score(y, s)
                    if np.unique(y).size > 1 else float("nan")
                )

            ret["MacroF1_over_categories"] = np.mean([ret[f"F1_cate_{c}"] for c in range(C)])
            ret["MacroAP_over_categories"] = np.nanmean([ret[f"AP_cate_{c}"] for c in range(C)])
            ret["MacroAcc_over_categories"] = np.nanmean([ret[f"Acc_cate_{c}"] for c in range(C)])

            # ---- sparsity ----
            if isSparsity:
                for k in (1, 2, 3, 4):
                    y = np.concatenate(per_mask_labels[k])
                    p = np.concatenate(per_mask_preds[k])
                    s = np.concatenate(per_mask_scores[k])

                    ret[f"F1_mask_{k}"] = f1_score(y, p, zero_division=0)
                    ret[f"Acc_mask_{k}"] = accuracy_score(y, p) if y.size > 0 else float("nan")
                    ret[f"AP_mask_{k}"] = (
                        average_precision_score(y, s)
                        if np.unique(y).size > 1 else float("nan")
                    )

                # ---- category × sparsity (4×4 table) ----
                for c in range(C):
                    for k in (1, 2, 3, 4):
                        y = np.concatenate(per_cat_mask_labels[(c, k)])
                        p = np.concatenate(per_cat_mask_preds[(c, k)])
                        s = np.concatenate(per_cat_mask_scores[(c, k)])

                        ret[f"F1_cate_{c}_mask_{k}"] = f1_score(y, p, zero_division=0)
                        ret[f"Acc_cate_{c}_mask_{k}"] = accuracy_score(y, p) if y.size > 0 else float("nan")
                        ret[f"AP_cate_{c}_mask_{k}"] = (
                            average_precision_score(y, s)
                            if np.unique(y).size > 1 else float("nan")
                        )

        if isSparsity == False:
            if want_error:
                for i in range(args.offNum):
                    ret['RMSE_%d' % i] = np.sqrt(epochSqLoss[i] / epochTstNum[i])
                    ret['MAE_%d' % i] = epochAbsLoss[i] / epochTstNum[i]
                    ret['MAPE_%d' % i] = epochApeLoss[i] / epochPosNums[i]
                ret['RMSE'] = np.sqrt(np.sum(epochSqLoss) / np.sum(epochTstNum))
                ret['MAE'] = np.sum(epochAbsLoss) / np.sum(epochTstNum)
                ret['MAPE'] = np.sum(epochApeLoss) / np.sum(epochPosNums)
                ret['epochLoss'] = epochLoss
        else:
            if want_error:
                ret['RMSE'] = np.sqrt(np.sum(epochSqLoss) / np.sum(epochTstNum))
                ret['MAE'] = np.sum(epochAbsLoss) / np.sum(epochTstNum)
                ret['MAPE'] = np.sum(epochApeLoss) / np.sum(epochPosNums)
                for k in range(args.offNum):
                    ret['RMSE_%d' % k] = np.sqrt(epochSqLoss[k] / epochTstNum[k])
                    ret['MAE_%d' % k] = epochAbsLoss[k] / epochTstNum[k]
                    ret['MAPE_%d' % k] = epochApeLoss[k] / epochPosNums[k]

                ret['RMSE_mask_1'] = np.sqrt(np.sum(epochSqLoss1) / np.sum(epochTstNum1))
                ret['MAE_mask_1'] = np.sum(epochAbsLoss1) / np.sum(epochTstNum1)
                ret['MAPE_mask_1'] = np.sum(epochApeLoss1) / np.sum(epochPosNums1)

                ret['RMSE_mask_2'] = np.sqrt(np.sum(epochSqLoss2) / np.sum(epochTstNum2))
                ret['MAE_mask_2'] = np.sum(epochAbsLoss2) / np.sum(epochTstNum2)
                ret['MAPE_mask_2'] = np.sum(epochApeLoss2) / np.sum(epochPosNums2)

                ret['RMSE_mask_3'] = np.sqrt(np.sum(epochSqLoss3) / np.sum(epochTstNum3))
                ret['MAE_mask_3'] = np.sum(epochAbsLoss3) / np.sum(epochTstNum3)
                ret['MAPE_mask_3'] = np.sum(epochApeLoss3) / np.sum(epochPosNums3)

                ret['RMSE_mask_4'] = np.sqrt(np.sum(epochSqLoss4) / np.sum(epochTstNum4))
                ret['MAE_mask_4'] = np.sum(epochAbsLoss4) / np.sum(epochTstNum4)
                ret['MAPE_mask_4'] = np.sum(epochApeLoss4) / np.sum(epochPosNums4)
                ret['epochLoss'] = epochLoss
        return ret


def sampleTestBatch(batIds, st, ed, tstTensor, inpTensor, handler):
    batch = ed - st
    idx = batIds[0: batch]
    label = tstTensor[:, idx, :]
    label = np.transpose(label, [1, 0, 2])
    retLabels = label
    mask = handler.tstLocs * (label > 0)

    feat_list = []
    for i in range(batch):
        if idx[i] - args.temporalRange < 0:
            temT = inpTensor[:, idx[i] - args.temporalRange:, :]
            temT2 = tstTensor[:, :idx[i], :]
            feat_one = np.concatenate([temT, temT2], axis=1)
        else:
            feat_one = tstTensor[:, idx[i] - args.temporalRange: idx[i], :]
        feat_one = np.expand_dims(feat_one, axis=0)
        feat_list.append(feat_one)
    feats = np.concatenate(feat_list, axis=0)
    return handler.zScore(feats), retLabels, mask,


def test(model, handler):
    want_error = args.eval_metrics in ("error", "all")
    want_acc   = args.eval_metrics in ("accuracy", "all")
    if args.eval_metrics == "none":
        want_error = False
        want_acc = False
    ids = np.array(list(range(handler.tstT.shape[1])))
    epochLoss, epochPreLoss, = [0] * 2
    epochSqLoss1, epochAbsLoss1, epochTstNum1, epochApeLoss1, epochPosNums1 = [np.zeros(4) for i in range(5)]
    epochSqLoss2, epochAbsLoss2, epochTstNum2, epochApeLoss2, epochPosNums2 = [np.zeros(4) for i in range(5)]
    epochSqLoss3, epochAbsLoss3, epochTstNum3, epochApeLoss3, epochPosNums3 = [np.zeros(4) for i in range(5)]
    epochSqLoss4, epochAbsLoss4, epochTstNum4, epochApeLoss4, epochPosNums4 = [np.zeros(4) for i in range(5)]
    epochSqLoss, epochAbsLoss, epochTstNum, epochApeLoss, epochPosNums = [np.zeros(4) for i in range(5)]
    num = len(ids)

    steps = int(np.ceil(num / args.batch))

    # ================= ACC METRIC CONTAINERS =================
    if want_acc:
        all_labels = []
        all_preds  = []
        all_scores = []

        C = args.offNum

        # per-category
        per_cat_labels = {c: [] for c in range(C)}
        per_cat_preds  = {c: [] for c in range(C)}
        per_cat_scores = {c: [] for c in range(C)}

        # per-sparsity
        per_mask_labels = {k: [] for k in (1, 2, 3, 4)}
        per_mask_preds  = {k: [] for k in (1, 2, 3, 4)}
        per_mask_scores = {k: [] for k in (1, 2, 3, 4)}

        # category × sparsity (4×4)
        per_cat_mask_labels = {(c, k): [] for c in range(C) for k in (1, 2, 3, 4)}
        per_cat_mask_preds  = {(c, k): [] for c in range(C) for k in (1, 2, 3, 4)}
        per_cat_mask_scores = {(c, k): [] for c in range(C) for k in (1, 2, 3, 4)}

    for i in range(steps):
        st = i * args.batch
        ed = min((i + 1) * args.batch, num)
        batIds = ids[st: ed]

        tem = sampleTestBatch(batIds, st, ed, handler.tstT, np.concatenate([handler.trnT, handler.valT], axis=1), handler)
        feats, labels, mask = tem
        feats = torch.Tensor(feats).to(args.device)
        idx = np.random.permutation(args.areaNum)
        shuf_feats = feats[:, idx, :, :]

        if args.use_ode_option == "baseline":
            out_local, eb_local, eb_global, DGI_pred, out_global = model(feats, shuf_feats)
            output = handler.zInverse(out_global)
        elif args.use_ode_option == "option1":
            out_local, eb_local, eb_global, DGI_pred, out_global, Z_t = model(feats, shuf_feats)
            output = handler.zInverse(out_global)
        elif args.use_ode_option == "option2":
            out_local, eb_local, eb_global, DGI_pred, out_global, fe, Z_t, pred = model(feats, shuf_feats)
            output = handler.zInverse(pred)
        else:
            raise ValueError("Unknown ODE option {args.use_ode_option}")

        if want_acc:
            lbl_bin  = (labels > args.f1_label_threshold).astype(int)
            pred_bin = (output.cpu().detach().numpy() > args.f1_pred_threshold).astype(int)
            scores   = output.cpu().detach().numpy()

            # overall
            all_labels.append(lbl_bin.reshape(-1))
            all_preds.append(pred_bin.reshape(-1))
            all_scores.append(scores.reshape(-1))

            # per-category
            for c in range(C):
                per_cat_labels[c].append(lbl_bin[:, :, c].reshape(-1))
                per_cat_preds[c].append(pred_bin[:, :, c].reshape(-1))
                per_cat_scores[c].append(scores[:, :, c].reshape(-1))

            # per-sparsity + category×sparsity
            masks = {
                1: handler.mask1.astype(bool),
                2: handler.mask2.astype(bool),
                3: handler.mask3.astype(bool),
                4: handler.mask4.astype(bool),
            }

            for k, mk in masks.items():
                for c in range(C):
                    a_mask = mk[:, c]
                    # category × mask (for 4×4 table)
                    per_cat_mask_labels[(c, k)].append(lbl_bin[:, :, c][:, a_mask].reshape(-1))
                    per_cat_mask_preds[(c, k)].append(pred_bin[:, :, c][:, a_mask].reshape(-1))
                    per_cat_mask_scores[(c, k)].append(scores[:, :, c][:, a_mask].reshape(-1))
                    per_mask_labels[k].append(lbl_bin[:, :, c][:, a_mask].reshape(-1))
                    per_mask_preds[k].append(pred_bin[:, :, c][:, a_mask].reshape(-1))
                    per_mask_scores[k].append(scores[:, :, c][:, a_mask].reshape(-1))

                    # [Bhavani] try this option instead of above 6 lines when GPU execution to see if there's difference
                    # a_mask = mk[:, c].astype(bool)     # shape: (A,)

                    # # Select exactly the regions belonging to this sparsity-bin for this category
                    # y_sel = lbl_bin[:, a_mask, c]      # shape: (B, Rk)
                    # p_sel = pred_bin[:, a_mask, c]
                    # s_sel = scores[:, a_mask, c]

                    # # Per-mask aggregates (across categories too, by appending category-wise)
                    # per_mask_labels[k].append(y_sel.reshape(-1))
                    # per_mask_preds[k].append(p_sel.reshape(-1))
                    # per_mask_scores[k].append(s_sel.reshape(-1))

                    # # Category × Mask aggregates
                    # per_cat_mask_labels[(c, k)].append(y_sel.reshape(-1))
                    # per_cat_mask_preds[(c, k)].append(p_sel.reshape(-1))
                    # per_cat_mask_scores[(c, k)].append(s_sel.reshape(-1))

        if want_error:
            _, sqLoss1, absLoss1, tstNums1, apeLoss1, posNums1 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, handler.mask1)
            _, sqLoss2, absLoss2, tstNums2, apeLoss2, posNums2 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, handler.mask2)
            _, sqLoss3, absLoss3, tstNums3, apeLoss3, posNums3 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, handler.mask3)
            _, sqLoss4, absLoss4, tstNums4, apeLoss4, posNums4 = utils.cal_metrics_r_mask(output.cpu().detach().numpy(), labels, mask, handler.mask4)

            epochSqLoss1 += sqLoss1
            epochAbsLoss1 += absLoss1
            epochTstNum1 += tstNums1
            epochApeLoss1 += apeLoss1
            epochPosNums1 += posNums1

            epochSqLoss2 += sqLoss2
            epochAbsLoss2 += absLoss2
            epochTstNum2 += tstNums2
            epochApeLoss2 += apeLoss2
            epochPosNums2 += posNums2

            epochSqLoss3 += sqLoss3
            epochAbsLoss3 += absLoss3
            epochTstNum3 += tstNums3
            epochApeLoss3 += apeLoss3
            epochPosNums3 += posNums3

            epochSqLoss4 += sqLoss4
            epochAbsLoss4 += absLoss4
            epochTstNum4 += tstNums4
            epochApeLoss4 += apeLoss4
            epochPosNums4 += posNums4

            loss, sqLoss, absLoss, tstNums, apeLoss, posNums = utils.cal_metrics_r(output.cpu().detach().numpy(), labels, mask)
            epochSqLoss += sqLoss
            epochAbsLoss += absLoss
            epochTstNum += tstNums
            epochApeLoss += apeLoss
            epochPosNums += posNums

            epochLoss += loss
            print('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, loss), end='\r')
    ret = dict()

    if want_acc:
        labels_flat = np.concatenate(all_labels)
        preds_flat  = np.concatenate(all_preds)
        scores_flat = np.concatenate(all_scores)

        ret["MacroF1"] = f1_score(labels_flat, preds_flat, average="macro")
        ret["MicroF1"] = f1_score(labels_flat, preds_flat, average="micro")
        ret["AP"] = average_precision_score(labels_flat, scores_flat)
        ret["Acc"] = accuracy_score(labels_flat, preds_flat)
        # ---- per-category ----
        for c in range(C):
            y = np.concatenate(per_cat_labels[c])
            p = np.concatenate(per_cat_preds[c])
            s = np.concatenate(per_cat_scores[c])

            ret[f"F1_cate_{c}"] = f1_score(y, p, zero_division=0)
            ret[f"Acc_cate_{c}"] = accuracy_score(y, p) if y.size > 0 else float("nan")
            ret[f"AP_cate_{c}"] = (
                average_precision_score(y, s)
                if np.unique(y).size > 1 else float("nan")
            )

        ret["MacroF1_over_categories"] = np.mean([ret[f"F1_cate_{c}"] for c in range(C)])
        ret["MacroAP_over_categories"] = np.nanmean([ret[f"AP_cate_{c}"] for c in range(C)])
        ret["MacroAcc_over_categories"] = np.nanmean([ret[f"Acc_cate_{c}"] for c in range(C)])

        # ---- sparsity ----
        for k in (1, 2, 3, 4):
            y = np.concatenate(per_mask_labels[k])
            p = np.concatenate(per_mask_preds[k])
            s = np.concatenate(per_mask_scores[k])

            ret[f"F1_mask_{k}"] = f1_score(y, p, zero_division=0)
            ret[f"Acc_mask_{k}"] = accuracy_score(y, p) if y.size > 0 else float("nan")
            ret[f"AP_mask_{k}"] = (
                average_precision_score(y, s)
                if np.unique(y).size > 1 else float("nan")
            )

        # ---- category × sparsity (4×4 table) ----
        for c in range(C):
            for k in (1, 2, 3, 4):
                y = np.concatenate(per_cat_mask_labels[(c, k)])
                p = np.concatenate(per_cat_mask_preds[(c, k)])
                s = np.concatenate(per_cat_mask_scores[(c, k)])

                ret[f"F1_cate_{c}_mask_{k}"] = f1_score(y, p, zero_division=0)
                ret[f"Acc_cate_{c}_mask_{k}"] = accuracy_score(y, p) if y.size > 0 else float("nan")
                ret[f"AP_cate_{c}_mask_{k}"] = (
                    average_precision_score(y, s)
                    if np.unique(y).size > 1 else float("nan")
                )

    if want_error:
        ret['RMSE'] = np.sqrt(np.sum(epochSqLoss) / np.sum(epochTstNum))
        ret['MAE'] = np.sum(epochAbsLoss) / np.sum(epochTstNum)
        ret['MAPE'] = np.sum(epochApeLoss) / np.sum(epochPosNums)

        for i in range(args.offNum):
            ret['RMSE_%d' % i] = np.sqrt(epochSqLoss[i] / epochTstNum[i])
            ret['MAE_%d' % i] = epochAbsLoss[i] / epochTstNum[i]
            ret['MAPE_%d' % i] = epochApeLoss[i] / epochPosNums[i]

        ret['RMSE_mask_1'] = np.sqrt(np.sum(epochSqLoss1) / np.sum(epochTstNum1))
        ret['MAE_mask_1'] = np.sum(epochAbsLoss1) / np.sum(epochTstNum1)
        ret['MAPE_mask_1'] = np.sum(epochApeLoss1) / np.sum(epochPosNums1)

        ret['RMSE_mask_2'] = np.sqrt(np.sum(epochSqLoss2) / np.sum(epochTstNum2))
        ret['MAE_mask_2'] = np.sum(epochAbsLoss2) / np.sum(epochTstNum2)
        ret['MAPE_mask_2'] = np.sum(epochApeLoss2) / np.sum(epochPosNums2)

        ret['RMSE_mask_3'] = np.sqrt(np.sum(epochSqLoss3) / np.sum(epochTstNum3))
        ret['MAE_mask_3'] = np.sum(epochAbsLoss3) / np.sum(epochTstNum3)
        ret['MAPE_mask_3'] = np.sum(epochApeLoss3) / np.sum(epochPosNums3)

        ret['RMSE_mask_4'] = np.sqrt(np.sum(epochSqLoss4) / np.sum(epochTstNum4))
        ret['MAE_mask_4'] = np.sum(epochAbsLoss4) / np.sum(epochTstNum4)
        ret['MAPE_mask_4'] = np.sum(epochApeLoss4) / np.sum(epochPosNums4)
        ret['epochLoss'] = epochLoss

    return ret