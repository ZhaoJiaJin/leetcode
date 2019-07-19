/*
 * @lc app=leetcode id=198 lang=golang
 *
 * [198] House Robber
 */

 /*
dp[n] = max(dp[n-1],dp[n-2]+num[n], dp[n-3]+nums[n])
 */

func max(a,b int) int {
	if a>b{
		return a
	}
	return b
}

func max3(a,b,c int) int {
	if a > b && a > c{
		return a
	}
	if b > a && b > c{
		return b
	}
	return c
}

func rob(nums []int) int {
	if len(nums) == 0{
		return 0
	}
	if len(nums) == 1{
		return nums[0]
	}

	ret := make([]int, len(nums)+1)
	ret[0] = 0
	ret[1] = nums[0]
	ret[2] = nums[1]
	for i:=2; i < len(nums); i ++{
		ret[i+1] = max3(ret[i], ret[i-1] + nums[i], ret[i-2]+nums[i])
	}
	fmt.Println(ret)
	return max(ret[len(nums)], ret[len(nums)-1])
    
}

