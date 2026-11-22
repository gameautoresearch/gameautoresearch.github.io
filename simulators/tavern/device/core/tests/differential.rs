//! Independent host build of the device rule source versus the pinned native oracle.
use tavern_device_core as port;

#[test]
fn full_eight_player_rules_views_rng_and_checkpoint_replay(){
    std::thread::Builder::new().stack_size(64<<20).spawn(||{
        oracle::guard::thread_started(64<<20);
        port::guard::thread_started(64<<20);
        for seed in [12,12345,45678,78901]{
            let cfg=r#"{"num_players":8,"content":"Implemented","turn_action_caps":[30],"max_actions_per_turn":30,"max_turns":100,"record_log":false}"#;
            let mut a=oracle::Game::new(seed,serde_json::from_str(cfg).unwrap());
            let mut b=port::Game::new(seed,serde_json::from_str(cfg).unwrap());
            let mut decisions=0;
            for _ in 0..3000{
                assert_eq!(serde_json::to_value(&a).unwrap(),serde_json::to_value(&b).unwrap());
                for p in 0..8{
                    assert_eq!(a.needs_action(p),b.needs_action(p));
                    assert_eq!(a.legal_mask(p),b.legal_mask(p));
                    let x=oracle::obs::observe(&a,p);let y=port::obs::observe(&b,p);
                    assert_eq!(x.floats,y.floats);assert_eq!(x.cards,y.cards);
                }
                if a.is_over(){break}
                let p=(0..8).find(|&p|a.needs_action(p)).expect("progressing ready seat");
                let x=oracle::bot::choose_action(&mut a,p).encode();
                let y=port::bot::choose_action(&mut b,p).encode();
                assert_eq!(x,y);
                assert_eq!(a.apply_index(p,x),b.apply_index(p,y));
                decisions+=1;
                if decisions%31==0{
                    let state=serde_json::to_string(&b).unwrap();
                    b=serde_json::from_str(&state).unwrap();b.check_loaded().unwrap();
                    let imported:oracle::Game=serde_json::from_str(&state).unwrap();
                    assert_eq!(serde_json::to_value(&a).unwrap(),serde_json::to_value(&imported).unwrap());
                }
            }
            assert!(a.is_over());assert!(decisions>80);
            assert_eq!(a.players.iter().map(|p|p.placement).collect::<Vec<_>>(),
                b.players.iter().map(|p|p.placement).collect::<Vec<_>>());
        }
    }).unwrap().join().unwrap();
}
