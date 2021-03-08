from nltk.corpus import names
from nltk.classify import apply_features
import nltk
import random


class GenderRecoginition:
    """ use nltk classfication to identify gender. """

    def gender_features(self, word):
        return {
            'first-letter': word[0],  # First letter
            'first2-letters': word[0:2],  # First 2 letters
            'first3-letters': word[0:3],  # First 3 letters
            'last-letter': word[-1],
            'last2-letters': word[-2:],
            'last3-letters': word[-3:],
        }

    def gender_identify(self, word, isPrint):
        # featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
        # train_set, test_set = featuresets[500:], featuresets[:500]

        labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in
                                                                                 names.words('female.txt')])
        random.shuffle(labeled_names)

        train_set = apply_features(self.gender_features, labeled_names[500:])
        test_set = apply_features(self.gender_features, labeled_names[:500])

        classifier = nltk.NaiveBayesClassifier.train(train_set)

        if isPrint:
            print("gender recognise accuracy is " + str(nltk.classify.accuracy(classifier, test_set)))

        return classifier.classify(self.gender_features(word))


if __name__ == '__main__':
    genderRec = GenderRecoginition()
    print(genderRec.gender_identify("Lucy Green", True))
