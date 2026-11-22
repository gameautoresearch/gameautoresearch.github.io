//! Portable deterministic stable ordering for device-owned game lists.
//! The pinned core's branchless pointer small-sort path rejects valid integer
//! orderings on NVPTX. This implementation uses indices and explicit branches;
//! equal keys retain their input order, including prior RNG shuffles.
extern crate alloc;
use alloc::vec::Vec;
use core::cmp::Ordering;

pub fn sort<T:Copy+Ord>(values:&mut[T]){sort_by(values,|a,b|a.cmp(b))}
pub fn sort_unstable<T:Copy+Ord>(values:&mut[T]){sort(values)}
pub fn sort_by_key<T:Copy,K:Ord,F:FnMut(&T)->K>(values:&mut[T],mut key:F){
    sort_by(values,|a,b|key(a).cmp(&key(b)))
}
fn sort_by<T:Copy,F:FnMut(&T,&T)->Ordering>(values:&mut[T],mut compare:F){
    let n=values.len();if n<2{return}
    if n<=32{
        for i in 1..n{
            let value=values[i];let mut j=i;
            while j>0&&compare(&value,&values[j-1])==Ordering::Less{
                values[j]=values[j-1];j-=1;
            }
            values[j]=value;
        }
        return
    }
    let mut scratch:Vec<T>=values.to_vec();let mut width=1usize;
    while width<n{
        let stride=width.saturating_mul(2);let mut start=0usize;
        while start<n{
            let middle=start.saturating_add(width).min(n);
            let end=start.saturating_add(stride).min(n);
            let(mut left,mut right)=(start,middle);
            for out in start..end{
                if left<middle&&(right==end||compare(&values[left],&values[right])!=Ordering::Greater){
                    scratch[out]=values[left];left+=1;
                }else{scratch[out]=values[right];right+=1}
            }
            start=end;
        }
        values.copy_from_slice(&scratch);width=stride;
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn matches_original_stable_order_across_sizes_and_ties(){
        let mut rng=113u64;
        for n in [0,1,2,7,16,31,32,33,63,127,256,2049]{
            for _ in 0..5{
                let mut a:Vec<(i32,usize)>=(0..n).map(|i|{
                    rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (((rng>>32)%11)as i32-5,i)
                }).collect();
                let mut b=a.clone();b.sort_by_key(|row|row.0);sort_by_key(&mut a,|row|row.0);
                assert_eq!(a,b);
                let mut c:Vec<_>=a.iter().rev().map(|row|row.0).collect();
                let mut d=c.clone();d.sort();sort(&mut c);assert_eq!(c,d);
            }
        }
    }
}
