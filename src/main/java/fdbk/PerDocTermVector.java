/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fdbk;

import java.util.HashMap;

/**
 *
 * @author Debasis
 */
public class PerDocTermVector {
    int docId;
    int sum_tf;
    float sim;  // similarity with query
    HashMap<String, RetrievedDocTermInfo> perDocStats; // term --> wts
    
    public PerDocTermVector(int docId) {
        this.docId = docId;
        perDocStats = new HashMap<>();
        sum_tf = 0;
    }
    
    public float getNormalizedTf(String term) {
        RetrievedDocTermInfo tInfo = perDocStats.get(term);
        if (tInfo == null)
            return 0;
        return perDocStats.get(term).tf/(float)sum_tf;
    }

    public float getTf(String term) {
        RetrievedDocTermInfo tInfo = perDocStats.get(term);
        if (tInfo == null)
            return 0;
        return perDocStats.get(term).tf;
    }

    public RetrievedDocTermInfo getTermStats(String qTerm) {
        return this.perDocStats.get(qTerm);
    }

    public HashMap<String, RetrievedDocTermInfo> getPerDocStats() { return perDocStats; }

    public void addTerm(String termText, int tf) {
        this.perDocStats.put(termText, new RetrievedDocTermInfo(termText, tf));
    }
}

