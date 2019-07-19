/*
 * @lc app=leetcode id=268 lang=golang
 *
 * [268] Missing Number
 *
 * https://leetcode.com/problems/missing-number/description/
 *
 * algorithms
 * Easy (48.79%)
 * Likes:    979
 * Dislikes: 1386
 * Total Accepted:    293.8K
 * Total Submissions: 602K
 * Testcase Example:  '[3,0,1]'
 *
 * Given an array containing n distinct numbers taken from 0, 1, 2, ..., n,
 * find the one that is missing from the array.
 * 
 * Example 1:
 * 
 * 
 * Input: [3,0,1]
 * Output: 2
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: [9,6,4,2,3,5,7,0,1]
 * Output: 8
 * 
 * 
 * Note:
 * Your algorithm should run in linear runtime complexity. Could you implement
 * it using only constant extra space complexity?
 */
func missingNumber(nums []int) int {
	a := make([]byte,(1<<31)/8)
	for _,v1 := range nums{
		v := uint(v1)
		a[v/8] = (a[v/8] | (1 << (v%8) )) 
	}
	var i uint
	for idx,v := range a{
		for i=0; i < 8; i ++{
			if v & (1 << i) == 0{
				return idx*8 + int(i) 
			}
		}
	}
	return 1<<63 - 1
}

