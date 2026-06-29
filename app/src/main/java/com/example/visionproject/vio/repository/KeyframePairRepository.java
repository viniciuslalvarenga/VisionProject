package com.example.visionproject.vio.repository;

import com.example.visionproject.vio.model.KeyframePair;

public class KeyframePairRepository {
    private static KeyframePairRepository instance;
    private KeyframePair pair;

    private KeyframePairRepository() {}

    public static synchronized KeyframePairRepository getInstance() {
        if (instance == null) instance = new KeyframePairRepository();
        return instance;
    }

    public synchronized void setPair(KeyframePair p) { this.pair = p; }
    public synchronized KeyframePair getPair()       { return pair; }
    public synchronized void clear()                 { pair = null; }
}
