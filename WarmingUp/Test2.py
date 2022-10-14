import numpy as np

list = [1, 2, 3, 4, 5]
list2 = [[9, 29, 45], [14, 32, 22]]
list2.append(list)

print(list[:1])
print(list[1:])
print(list[0::2])


class lists:
    def __init__(self, list):
        self.list = list

    @property
    def plus_two(self):
        lst_2 = []
        for i in list:
            lst_2.append(i+2)

        return lst_2

    @property
    def minus_three(self):
        lst_3 = []
        for i in list:
            lst_3.append(i - 3)
        return lst_3


@property
def plus_hundred(self):
    list_4 = []
    for i in list:
        list_4.append(i + 100)


lists.plus_hundred = plus_hundred


x = lists(list)
print(x.plus_two)
print(x.minus_three)
# z = lists.plus_hundred
# print(x.minus_three)

z = np.array(list2)
print(list2)
z = z.reshape(1,3)
print(z.shape)
