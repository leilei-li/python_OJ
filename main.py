from leetcode_1 import Solution
from extra_class import *


def main():
    s = Solution()
    # s.findMedianSortedArrays([1, 2], [3,4])
    l1 = ListNode(1)
    l1.next = ListNode(2)
    l1.next.next = ListNode(4)
    l2 = ListNode(1)
    l2.next = ListNode(3)
    l2.next.next = ListNode(4)
    k = s.mergeTwoLists(l1, l2)
    while k is not None:
        print(k.val)
        k = k.next


if __name__ == '__main__':
    main()
