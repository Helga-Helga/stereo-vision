from numpy import inf
from Semiring.SemiringElement import SemiringElement


class SemiringArgminPlusElement(SemiringElement):
    def __init__(self, value=None):
        if value is not None:
            self.value = value
        super(SemiringArgminPlusElement, self).__init__(value)

    def mul(self, element):
        self.value = self.value + element.value
        return self

    def add(self, element, next_label, label_r):
        if self.value > element.value:
            self.value = element.value
            next_label = label_r
        return self, next_label

    @staticmethod
    def get_zero():
        return SemiringArgminPlusElement(inf)

    @staticmethod
    def get_unity():
        return SemiringArgminPlusElement(0)
