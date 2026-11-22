//! Shared deterministic map policy: no OS entropy or host RNG in device rules.
use core::{hash::{BuildHasher,BuildHasherDefault,Hash},ops::{Deref,DerefMut}};
type Fx = BuildHasherDefault<crate::cards::FxHasher>;
#[derive(Clone)]
pub struct HashMap<K,V,S=Fx>(hashbrown::HashMap<K,V,S>);
impl<K:Eq+Hash,V> HashMap<K,V,Fx>{
    pub fn new()->Self{Self(hashbrown::HashMap::with_hasher(Fx::default()))}
    pub fn with_capacity(n:usize)->Self{Self(hashbrown::HashMap::with_capacity_and_hasher(n,Fx::default()))}
}
impl<K,V,S> HashMap<K,V,S>{
    pub fn with_capacity_and_hasher(n:usize,s:S)->Self{Self(hashbrown::HashMap::with_capacity_and_hasher(n,s))}
}
impl<K:Eq+Hash,V,S:Default+BuildHasher> Default for HashMap<K,V,S>{
    fn default()->Self{Self(hashbrown::HashMap::with_hasher(S::default()))}
}
impl<K,V,S> Deref for HashMap<K,V,S>{type Target=hashbrown::HashMap<K,V,S>;fn deref(&self)->&Self::Target{&self.0}}
impl<K,V,S> DerefMut for HashMap<K,V,S>{fn deref_mut(&mut self)->&mut Self::Target{&mut self.0}}
impl<K:core::fmt::Debug,V:core::fmt::Debug,S> core::fmt::Debug for HashMap<K,V,S>{
    fn fmt(&self,f:&mut core::fmt::Formatter<'_>)->core::fmt::Result{self.0.fmt(f)}
}
