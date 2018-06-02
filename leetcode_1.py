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

    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if abs(x) == x:
            flag = 1
        else:
            flag = 0
        result = 0
        x = abs(x)
        while x != 0:
            k = x % 10
            result = result * 10 + k
            x = int(x / 10)
        if abs(result) >= 2 ** 31 - 1:
            return 0
        if flag == 1:
            return result
        if flag == 0:
            return 0 - result

    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        str = str.strip()
        result = ''
        if str == '':
            return 0
        flag = 1
        if str[0] == '-':
            flag = 0
        elif str[0] == '+':
            pass
        else:
            try:
                int(str[0])
                result = result + str[0]
            except:
                return 0
        for i in range(1, len(str)):
            try:
                int(str[i])
            except:
                break
            result = result + str[i]
        try:
            res = int(result)
        except:
            return 0
        if flag == 0:
            res = 0 - res
            if res <= -2 ** 31:
                return -2 ** 31
            return res
        if flag == 1:
            res = res
            if res >= 2 ** 31 - 1:
                return 2 ** 31 - 1
            return res

    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        if x >= 0 and x <= 9:
            return True
        palind = 0
        source = x
        while x > 0:
            k = x % 10
            palind = palind * 10 + k
            x = int(x / 10)
        if palind == source:
            return True
        return False

    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        dp = []
        for i in range(len(s) + 1):
            dp.append(False)
        dp[len(s)] = True
        i = len(p) - 1
        while i >= 0:
            if p[i] == '*':
                for j in range(len(s) - 1, -1, -1):
                    dp[j] = dp[j] or (dp[j + 1] and (p[i - 1] == '.' or p[i - 1] == s[j]))
                i = i - 1
            else:
                for j in range(len(s)):
                    dp[j] = dp[j + 1] and (p[i] == '.' or p[i] == s[j])
                dp[len(s)] = False
            i = i - 1
        return dp[0]

    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        area = 0
        left = 0
        right = len(height) - 1
        while left < right:
            s = min(height[left], height[right]) * (right - left)
            area = max(area, s)
            if height[left] > height[right]:
                right = right - 1
            elif height[left] <= height[right]:
                left = left + 1
        return area

    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        M = ["", "M", "MM", "MMM"]
        C = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
        X = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
        I = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
        return M[int(num / 1000)] + C[int((num % 1000) / 100)] + X[int((num % 100) / 10)] + I[num % 10]

    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = {}
        d['I'] = 1
        d['V'] = 5
        d['X'] = 10
        d['L'] = 50
        d['C'] = 100
        d['D'] = 500
        d['M'] = 1000
        result = 0
        pre = 0
        for i in range(len(s) - 1, -1, -1):
            cur = d[s[i]]
            if cur < pre:
                result = result - cur
            else:
                result = result + cur
            pre = cur
        return result

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        pre_fix = ''
        min_str = ''
        min_len = 9999
        for str in strs:
            if len(str) >= min_len:
                min_len = len(str)
            min_str = str
        while min_str != '':
            flag = 1
            for str in strs:
                cur_str = str[:len(min_str)]
                if cur_str == min_str:
                    continue
                else:
                    min_str = min_str[:len(min_str) - 1]
                    flag = 0
                    break
            if flag == 1:
                return min_str
        return ''

        return pre_fix

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        size = len(nums)
        ans = []
        if size <= 2:
            return ans
        nums.sort()
        i = 0
        while i < size - 2:
            tmp = 0 - nums[i]
            j = i + 1
            k = size - 1
            while j < k:
                if nums[j] + nums[k] < tmp:
                    j += 1
                elif nums[j] + nums[k] > tmp:
                    k -= 1
                else:
                    ans.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1
                    while j < k:
                        if nums[j] != nums[j - 1]:
                            break
                        if nums[k] != nums[k + 1]:
                            break
                        j += 1
                        k -= 1
            i += 1
            while i < size - 2:
                if nums[i] != nums[i - 1]:
                    break
                i += 1
        return ans

    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        error = 9999
        closest = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    sum = nums[i] + nums[j] + nums[k]
                    if abs(sum - target) <= error:
                        error = abs(sum - target)
                        closest = sum
        return closest

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if len(digits) == 0:
            return []
        result = []
        d = {}
        d['0'] = [' ']
        d['2'] = ['a', 'b', 'c']
        d['3'] = ['d', 'e', 'f']
        d['4'] = ['g', 'h', 'i']
        d['5'] = ['j', 'k', 'l']
        d['6'] = ['m', 'n', 'o']
        d['7'] = ['p', 'q', 'r', 's']
        d['8'] = ['t', 'u', 'v']
        d['9'] = ['w', 'x', 'y', 'z']
        self.__get_phone_str(digits, 0, result, '', d)
        return result

    def __get_phone_str(self, digits, position, result, str, d):
        if position < len(digits):
            if digits[position] in d.keys():
                for c in d[digits[position]]:
                    self.__get_phone_str(digits, position + 1, result, str + c, d)
        else:
            result.append(str)

    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        result = []
        nums.sort()
        for i in range(len(nums)):
            three_sum = target - nums[i]
            left_list = nums[i + 1:]
            three_sum_result = self.__threeSum_four_sum(left_list, three_sum)
            for l in three_sum_result:
                if [nums[i]] + l not in result:
                    result.append([nums[i]] + l)
        return result

    def __threeSum_four_sum(self, nums, target):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        size = len(nums)
        ans = []
        if size <= 2:
            return ans
        i = 0
        while i < size - 2:
            tmp = target - nums[i]
            j = i + 1
            k = size - 1
            while j < k:
                if nums[j] + nums[k] < tmp:
                    j += 1
                elif nums[j] + nums[k] > tmp:
                    k -= 1
                else:
                    ans.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1
                    while j < k:
                        if nums[j] != nums[j - 1]:
                            break
                        if nums[k] != nums[k + 1]:
                            break
                        j += 1
                        k -= 1
            i += 1
            while i < size - 2:
                if nums[i] != nums[i - 1]:
                    break
                i += 1
        return ans
