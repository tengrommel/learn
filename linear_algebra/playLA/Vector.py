class Vector:
    def __init__(self, lst):
        self._values = list(lst)

    def __sub__(self, another):
        """
        向量减法，返回结果向量
        :param another:
        :return:
        """
        assert len(self) == len(another), "Error in subtracting. Length of vectors must be same."
        return Vector([a-b for a, b in zip(self, another)])

    def __add__(self, another):
        """
        向量加法，返回结果向量
        :param another:
        :return:
        """
        assert len(self) == len(another), "Error in adding. Length of vectors must be same."
        return Vector([a+b for a, b in zip(self, another)])

    def __rmul__(self, k):
        """返回数量乘法的结果"""
        return self * k

    def __mul__(self, k):
        """
        返回数量乘法的结果向量
        :param k:
        :return:
        """
        return Vector([k * e for e in self])

    def __iter__(self):
        """
        返回向量的迭代器
        :return:
        """
        return self._values.__iter__()

    def __getitem__(self, index):
        """
        取出向量的第index个元素
        :param index:
        :return:
        """
        return self._values[index]

    def __len__(self):
        """
        返回向量的长度
        :return:
        """
        return len(self._values)

    def __repr__(self):
        return "Vector({})".format(self._values)

    def __str__(self):
        return "({})".format(", ".join(str(e) for e in self._values))
