/*
 * @lc app=leetcode id=202 lang=golang
 *
 * [202] Happy Number
 */

func cal(n int)int{
	res := 0
	for n != 0{
		left := n%10
		res += (left*left)
		n = n/10
	}
	return res
}

func isHappy(n int) bool {
    if n <= 0{
		return false
	}
	m:=make(map[int]bool)
	for{
		if n == 1{
			return true
		}
		
		n = cal(n)
		if _,ok := m[n]; ok{
			break
		}
		m[n] = true
		
	}
	return false
}

