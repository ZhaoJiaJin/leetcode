package main

import(
	"fmt"
)

type LinkList struct{
	Val int
	Key int
	Pre *LinkList
	Next *LinkList
}



type LRUCache struct {
	m map[int]*LinkList
	size int
	capa int
	head *LinkList
	tail *LinkList
}


func Constructor(capacity int) LRUCache {
	head := &LinkList{}
	tail := &LinkList{}
	head.Next = tail
	tail.Pre = head
	return LRUCache{m:make(map[int]*LinkList),size:0,capa:capacity,head:head,tail:tail}
    
}


func (this *LRUCache) Get(key int) int {
	node,present := this.m[key]
	if !present{
		return -1
	}
	node.Pre.Next = node.Next
	node.Next.Pre = node.Pre
	this.tail.Pre.Next = node
	node.Next = this.tail
	node.Pre = this.tail.Pre
	this.tail.Pre = node
	this.Print()
	return node.Val
}


func (this *LRUCache) Put(key int, value int)  {
	node,present := this.m[key]
	if present{
		node.Val = value
	node.Pre.Next = node.Next
	node.Next.Pre = node.Pre
	this.tail.Pre.Next = node
	node.Next = this.tail
	node.Pre = this.tail.Pre
	this.tail.Pre = node
	this.Print()

		return
	}
	if this.size >= this.capa{
		least := this.head.Next
		this.head.Next = this.head.Next.Next
		this.head.Next.Pre = this.head
		fmt.Println("evicts",least.Key)
		delete(this.m,least.Key)
		this.size --
	}
	node = &LinkList{Val:value,Key:key}
	this.m[key] = node
	this.tail.Pre.Next = node
	node.Next = this.tail
	node.Pre = this.tail.Pre
	this.tail.Pre = node
	this.size ++
	this.Print()
}


func (this *LRUCache) Print()  {
	n := this.head.Next
	for n != this.tail{
		fmt.Printf("%d %s",n.Key,"--> ")
		n = n.Next
	}
	fmt.Println()

}


func main(){
	cache := Constructor(2)

	cache.Put(2, 1);
	cache.Put(1, 1);
	cache.Put(2, 3);
	cache.Put(4, 1);
	fmt.Println(cache.Get(1));       
	fmt.Println(cache.Get(2));       

}

/**
 * Your LRUCache object will be instantiated and called as such:
 * obj := Constructor(capacity);
 * param_1 := obj.Get(key);
 * obj.Put(key,value);
 */
