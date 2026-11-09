# GoEmotions paper notes — starter draft

**Source:** Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G.,
& Ravi, S. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*.
Proceedings of ACL, 4040–4054. <https://aclanthology.org/2020.acl-main.372/>

The paper introduces GoEmotions, a collection of about 58,000 English Reddit
comments labeled for 27 emotion categories plus neutral. The authors motivate
fine-grained labels because emotion in language is not limited to a simple
positive-versus-negative distinction. The dataset was manually annotated, so
the labels represent how people perceived each comment. They are not direct
evidence of what the writer privately felt.

The authors evaluate emotion-classification models and report that their
BERT-based model achieves an average F1 score of about 0.46 across the
fine-grained taxonomy. This leaves substantial room for improvement and makes
the dataset useful for studying model errors and label difficulty, not just
headline accuracy.

For this project, the simplified train/dev/test files support model training
and evaluation, while the raw rater-level files make it possible to measure
how much annotators agreed on each comment. That distinction matters for H5:
comments with no agreed simplified label are still informative about human
disagreement. The project should describe model outputs as predicted or
perceived emotion, not as the author's true emotion.

The dataset also limits what can be concluded. It contains English Reddit
comments rather than a representative sample of people or conversations.
Comments are short, and the dataset's curation and label distribution affect
what the model can learn. The project's results therefore describe this
dataset, not Reddit as a whole; observational associations do not establish
causes.

**Before treating these as final notes:** read the paper itself and add or
correct details about the annotation procedure, sampling, quality checks, and
limitations. This is an AI-assisted study draft, not a record of personal
reading.
