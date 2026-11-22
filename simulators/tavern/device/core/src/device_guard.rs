//! Device trail instrumentation. Game's combat/effect cutoffs stay in the shared rules.
//! The CPU OS-stack diagnostic is not a CUDA stack checker. Device stack capacity is
//! separately set/recorded by the launcher and must pass difficult-mechanic parity.
use crate::engine::Ctx;
pub const TRAIL:usize=64;
pub fn floor()->usize{0}
pub fn thread_started(_:usize){}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Site {
    Event,
    Summon,
    Deathrattles,
    Deaths,
    Damage,
    Battlecry,
    TriggeredBattlecry,
    AddToHand,
    GiveCard,
    Triples,
    Cast,
    Discover,
    Magnetize,
    Attack,
    AttackNow,
}

impl Ctx<'_> {
    #[inline(always)]
    pub fn guard(&mut self,site:Site,arg:u32){
        self.trail[self.trail_n as usize%TRAIL]=(site as u8,arg);
        self.trail_n=self.trail_n.wrapping_add(1);
    }
}
