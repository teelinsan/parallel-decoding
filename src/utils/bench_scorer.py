from src.utils.bleu_calculator import BleuEvaluator, BleuValues


class Scorer(object):
    def __init__(self, name, acronym):
        self.name = name
        self.acronym = acronym

        self.bleu_scorer = BleuEvaluator()

        # number of sentences
        self.i = 0

        # param bleu score
        self.predictions = []
        self.references = []

        # benchmark values
        self.tot_mean_time = 0
        self.tot_mean_iter = 0

        # inline values
        self.current_init = None
        self.current_transl = None
        self.current_time = None
        self.current_iter = None

    def update_metrics(self, time, iter, translation, gold, init):
        self.tot_mean_time += (time - self.tot_mean_time) / (self.i + 1)
        self.tot_mean_iter += (iter - self.tot_mean_iter) / (self.i + 1)

        self.predictions.append(translation)
        self.references.append([gold])

        self.current_init = init
        self.current_transl = translation
        self.current_time = time
        self.current_iter = iter

        self.i += 1

    def compute_bleu_score(self):
        bleu_score = self.bleu_scorer.final_score(
            model_predictions=self.predictions,
            gold_references=self.references
        )

        bleu_dict = {
            'score': bleu_score['score'],
            'counts': str(bleu_score['counts']),
            'totals': str(bleu_score['totals']),
            'precisions': str(['%.2f' % prec for prec in bleu_score['precisions']]),
            'bp': bleu_score['bp'],
            'sys_len': bleu_score['sys_len'],
            'ref_len': bleu_score['ref_len']
        }

        return BleuValues(**bleu_dict)