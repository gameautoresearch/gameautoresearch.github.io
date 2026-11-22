//! Compact copy-on-write storage for infrequently changed minion data.
//!
//! Empty values occupy one pointer and allocate nothing. Combat/scouting clones
//! share immutable payloads; the first write detaches, preserving value semantics.
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub struct SharedVec<T: Clone + 'static>(Option<Arc<Vec<T>>>);
impl<T: Clone> Default for SharedVec<T> {
    fn default() -> Self {
        Self(None)
    }
}
impl<T: Clone> Clone for SharedVec<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T: Clone> Deref for SharedVec<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        self.0.as_deref().unwrap_or(const { &Vec::new() })
    }
}
impl<T: Clone> DerefMut for SharedVec<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        Arc::make_mut(self.0.get_or_insert_with(|| Arc::new(Vec::new())))
    }
}
impl<T: Clone> From<Vec<T>> for SharedVec<T> {
    fn from(value: Vec<T>) -> Self {
        if value.is_empty() {
            Self(None)
        } else {
            Self(Some(Arc::new(value)))
        }
    }
}
impl<T: Clone> SharedVec<T> {
    pub fn into_vec(self) -> Vec<T> {
        self.0.map_or_else(Vec::new, |a| {
            Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())
        })
    }
    pub fn clear(&mut self) {
        self.0 = None
    }
    pub fn shared_arc(&self) -> Arc<Vec<T>> {
        self.0.clone().unwrap_or_else(|| Arc::new(Vec::new()))
    }
}
impl<T: Clone + PartialEq> PartialEq for SharedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl<T: Clone + PartialEq> PartialEq<Vec<T>> for SharedVec<T> {
    fn eq(&self, other: &Vec<T>) -> bool {
        **self == *other
    }
}
impl<T: Clone + Eq> Eq for SharedVec<T> {}
impl<T: Clone + std::fmt::Debug> std::fmt::Debug for SharedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}
impl<T: Clone + Serialize> Serialize for SharedVec<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (**self).serialize(s)
    }
}
impl<'de, T: Clone + Deserialize<'de>> Deserialize<'de> for SharedVec<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Vec::<T>::deserialize(d).map(Self::from)
    }
}
impl<T: Clone> IntoIterator for SharedVec<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}
impl<'a, T: Clone> IntoIterator for &'a SharedVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a, T: Clone> IntoIterator for &'a mut SharedVec<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
impl<T: Clone> FromIterator<T> for SharedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Self {
        Vec::from_iter(it).into()
    }
}
impl<T: Clone> Extend<T> for SharedVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, it: I) {
        (**self).extend(it)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clones_detach_and_serialization_keeps_vector_schema() {
        let mut a: SharedVec<u32> = vec![1, 2, 3].into();
        let mut b = a.clone();
        b.push(4);
        a[0] = 9;
        assert_eq!(&**a, &[9, 2, 3]);
        assert_eq!(&**b, &[1, 2, 3, 4]);
        assert_eq!(serde_json::to_string(&b).unwrap(), "[1,2,3,4]");
        assert_eq!(
            serde_json::from_str::<SharedVec<u32>>("[1,2,3,4]").unwrap(),
            b
        );
        assert_eq!(
            std::mem::size_of::<SharedVec<u32>>(),
            std::mem::size_of::<usize>()
        );
    }
}
