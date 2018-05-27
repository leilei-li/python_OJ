class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap = {}
        for i in range(len(nums)):
            if nums[i] in hashmap.keys():
                return [hashmap[nums[i]], i]
                break
            else:
                hashmap[target - nums[i]] = i

    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        else:
            carry = 0
            ret = ListNode(0)
            ret_Last = ret
            while (l1 or l2):
                sum = 0
                if (l1):
                    sum = l1.val
                    l1 = l1.next
                if (l2):
                    sum += l2.val
                    l2 = l2.next
                sum += carry
                ret_Last.next = ListNode(sum % 10)
                ret_Last = ret_Last.next
                carry = (sum >= 10)
            if (carry):
                ret_Last.next = ListNode(1)
            ret_Last = ret.next
            del ret
            return ret_Last

    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == 0: return 0
        left_bound = 0
        max_value = 0
        hash_map = {}
        for i in range(len(s)):
            c = s[i]
            is_same = 0
            if c in hash_map.keys():
                is_same = hash_map[c] + 1
            left_bound = max(left_bound, is_same)
            max_value = max(max_value, i - left_bound + 1)
            hash_map[c] = i
        return max_value

    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        num3 = nums1 + nums2
        num3 = sorted(num3)
        l = len(num3)
        if l % 2 == 0:
            return float((num3[int((l - 1) / 2)] + num3[int(l / 2)]) / 2)
        else:
            return float(num3[int((l) / 2)])

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        self.start_index = 0
        self.length = 0
        if len(s) < 2:
            return s
        for i in range(len(s)):
            self.__findLongestPalindrome(s, i, i)
            self.__findLongestPalindrome(s, i, i + 1)
        return s[self.start_index:self.start_index + self.length]

    def __findLongestPalindrome(self, s, start, end):
        while start >= 0 and end < len(s) and s[start] == s[end]:
            start = start - 1
            end = end + 1
        if self.length < end - start - 1:
            self.start_index = start + 1
            self.length = end - start - 1

    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        result = []
        if len(s) < 0 or len(s) < numRows:
            return s
        for i in range(len(s)):
            result.append([])
        index = 0
        while index < len(s):
            for i in range(numRows):
                if index >= len(s):
                    break
                result[i].append(s[index])
                index = index + 1
                # 处理竖线
            for i in range(numRows - 2, 0, -1):
                if index >= len(s):
                    break
                result[i].append(s[index])
                index = index + 1
                # 处理斜线
        r = ''
        for i in range(numRows):
            k = ''.join(result[i])
            r = r + k
        return r
