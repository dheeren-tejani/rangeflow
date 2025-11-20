import math
from itertools import product

class Ranges:
    def __init__(self, *args):
        self.values: tuple = args
        self.range_template: str = "[{} <-----> {}]"
        self.ranges_list: list = []

    def range_adder(self):
        if len(self.values) % 2 == 0:
            value1: float = sum(self.values[i] for i in range(0, len(self.values), 2))  
            value2: float  = sum(self.values[i] for i in range(1, len(self.values), 2))
            return value1, value2
        else:
            print("Error: Dimension Issue, the count of values should be in even number")
                
    def range_subtractor(self):
        if len(self.values) % 2 == 0:
            value1: float = self.values[0]
            value2: float = self.values[1]

            for i in range(2, len(self.values), 2):
                value1 -= self.values[i]

            for i in range(3, len(self.values), 2):
                value2 -= self.values[i]
            return value1, value2
        else:
            print("Error: Dimension Issue, the count of values should be in even number")
        
    def range_multiplier(self):
        if len(self.values) % 2 == 0:
            single_ranges: list = [self.values[i:i+5] for i in range(0, len(self.values), 2)]
            result = 0

            for combo in product(*single_ranges):
                term = 1
                for x in combo:
                    term *= x
                result += term
            return result
        else:
            print("Error: Dimension Issue, the count of values should be in even number")

    def range_divisor(self):
        if len(self.values) % 2 == 0:
            ...
        else:
            print("Error: Dimension Issue, the count of values should be in even number")

    def range_pow(self):
        if len(self.values) % 2 == 0:
            ...
        else:
            print("Error: Dimension Issue, the count of values should be in even number")

    def range_root(self):
        if len(self.values) % 2 == 0:
            ...
        else:
            print("Error: Dimension Issue, the count of values should be in even number")

    def range_abs(self):
        if len(self.values) % 2 == 0:
            ...
        else:
            print("Error: Dimension Issue, the count of values should be in even number")

    def range_decay(self):
        if len(self.values) >= 4 and len(self.values) % 2 == 0:
            ...
        else:
            print("Error: Dimension Issue, the count of values should be in even number and Range should be in second Dimension or more")

    def range_step_decay(self, dimension) -> int:
        if dimension > 1:
            ...
        else:
            print("Error: Can't decay beyond a single dimensional range")

    def dimension_increaser(self) -> str:
        if len(self.values) % 2 == 0:
            current = self.range_template
            for i in range(1, int(math.log2(len(self.values)))):
                num_placeholders = 2 ** i
                current = current.format(*[self.range_template] * num_placeholders)
            current = current.format(*self.values)
            return current

        else:
            print("Error: Dimension Issue, the count of values should be in even number")


    def range_builder(self, value1, value2):
        return self.range_template.format(value1, value2)


if __name__ == '__main__':
    val1: list = [i for i in range(10000)]
    val2: list = [i for i in range(1, 20000, 2)]
    val3: list = [i for i in range(1, 30000, 3)]
    val4: list = [i for i in range(1, 40000, 4)]
    for i, j, k, l in zip(val1, val2, val3, val4):
        r = Ranges(i, j, k, l)
        value1, value2 = r.range_adder()
        print(r.range_builder(value1, value2))