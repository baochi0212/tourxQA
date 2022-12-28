from pprint import pprint

from model.IDSF_modules import *
import torch
import torch.nn as nn
from main import args
from transformers import AutoConfig, DistilBertConfig
from utils import MODEL_DICT, get_intent_labels, get_slot_labels, load_tokenizer, compute_metrics

import torch.nn as nn
import torch.nn.functional as F



class VanillaKD:
    """
    temperature-scaling KD
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer_teacher,
        optimizer_student,
        loss_fn=nn.MSELoss(),
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        super(VanillaKD, self).__init__(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            loss_fn,
            temp,
            distil_weight,
            device,
            log,
            logdir,
        )

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """

        soft_teacher_out = F.softmax(y_pred_teacher / self.temp, dim=1)
        soft_student_out = F.softmax(y_pred_student / self.temp, dim=1)

        loss = (1 - self.distil_weight) * F.cross_entropy(y_pred_student, y_true)
        loss += (self.distil_weight * self.temp * self.temp) * self.loss_fn(
            soft_teacher_out, soft_student_out
        )

        return loss


if __name__ == '__main__':
    teacher_config = AutoConfig.from_pretrained('vinai/phobert-base')
    student_config = DistilBertConfig.from_pretrained("bert-base-cased")
    intent_label_lst, slot_label_lst = get_intent_labels(args), get_slot_labels(args)
    teacher_model = JointPhoBERT(teacher_config, args, intent_label_lst, slot_label_lst)
    student_model = JointDistillBERT(student_config, args, intent_label_lst, slot_label_lst)
    pprint(teacher_model)
    pprint(student_model)
    