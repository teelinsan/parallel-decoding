from typing import Tuple

import numpy as np
import plotly.express as px
import torch
from transformers import MBart50Tokenizer


class DecodingDependencyGraph(object):
    def __init__(self, model, tokenizer, gold_target: torch.Tensor):
        self.model = model
        self.tokenizer = tokenizer
        self.init_matrix = []
        self.output_probs = []
        self.gold_target = gold_target

        if isinstance(tokenizer, MBart50Tokenizer):
            self.is_mbart = True
            self.starting_index = 2
        else:
            self.is_mbart = False
            self.starting_index = 1

    def insert_one_element(
        self, current_tensor, gold_tensor, index=None, output_probs=None
    ):
        self.init_matrix.append(
            (
                current_tensor[:, self.starting_index :]
                == gold_tensor[:, self.starting_index :]
            )
        )
        if output_probs is not None:
            self.output_probs.append(output_probs)

    def finalize_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.stack(self.init_matrix).permute(1, 0, 2).cpu(),
            torch.stack(self.output_probs).permute(1, 0, 2).cpu(),
        )

    def _create_labels(self, sample_index: int, method):
        sample_target = self.gold_target[sample_index, :].squeeze(0)

        if method == "decoded_ids":
            labels = self.tokenizer.convert_ids_to_tokens(sample_target)
            labels = [
                f"{i}:{id}"
                for i, id in zip(
                    labels[self.starting_index - 1 :],
                    sample_target[self.starting_index - 1 :],
                )
            ]
        elif method == "basic":
            labels = [f"{i}" for i in sample_target[self.starting_index - 1 :].tolist()]

        return labels

    def pretty_print(
        self, sample_index: int, sentence_id: str, method="basic", x_remap="text"
    ):
        labels = self._create_labels(sample_index=sample_index, method=method)
        iteration_matrix, probability_matrix = self.finalize_matrices()
        iteration_matrix = iteration_matrix[sample_index, :, :].int().numpy()
        probability_matrix = probability_matrix[sample_index, :, 1:].numpy()

        mask: np.ndarray = np.zeros_like(iteration_matrix)
        i, j = 0, 0
        while i < iteration_matrix.shape[0] and j < iteration_matrix.shape[1]:
            if iteration_matrix[i, j]:
                mask[i, j] += 1
                j += 1
            else:
                mask[i, j:] = iteration_matrix[i, j:]
                i += 1

        probability_matrix = mask * probability_matrix

        fig = px.imshow(
            iteration_matrix + mask,
            binary_compression_level=0,
            title=f"Decoding Dependency Graph for sentence {sentence_id}",
            color_continuous_scale="Viridis",
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(labels[1:]))),
            ticktext=[
                self.tokenizer.convert_ids_to_tokens([x])[0] if x_remap else str(x)
                for x in labels[1:]
            ],
            tickangle=45,
        )

        fig.update_traces(
            text=[
                [f"{xy:.2f}" if xy > 0 else "" for xy in x] for x in probability_matrix
            ],
            texttemplate="%{text}",
        )

        fig.update_layout(
            font=dict(family="Courier New, monospace", size=22, color="Black"),
            showlegend=False,
            coloraxis_showscale=False,
        )

        fig.show()

    def plot_confusion_matrix(
        self, cm, target_names, title="Confusion matrix", cmap=None, normalize=True
    ):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                                  the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                                  see http://matplotlib.org/examples/color/colormaps_reference.html
                                  plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                                  If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                                                                          # sklearn.metrics.confusion_matrix
                                                  normalize    = True,                # show proportions
                                                  target_names = y_labels_vals,       # list of names of the classes
                                                  title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        import itertools

        import matplotlib.pyplot as plt
        import numpy as np

        accuracy = np.trace(cm) / np.sum(cm).astype("float")
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap("Blues")

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(
                    j,
                    i,
                    "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
            else:
                plt.text(
                    j,
                    i,
                    "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel(
            "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(
                accuracy, misclass
            )
        )
        plt.show()
