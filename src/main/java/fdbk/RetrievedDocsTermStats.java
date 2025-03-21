/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package fdbk;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.BytesRef;
import retrieval.Constants;

/**
 *
 * @author Debasis
 */
public class RetrievedDocsTermStats {
    TopDocs topDocs;
    IndexReader reader;
    int sumTf;
    float sumDf;
    float sumSim;
    Map<String, RetrievedDocTermInfo> termStats; // stats for each term in top-k
    Map<Integer, PerDocTermVector> docTermVecs;
    int numTopDocs;
    
    public RetrievedDocsTermStats(IndexReader reader,
            TopDocs topDocs, int numTopDocs) {
        this.topDocs = topDocs;
        this.reader = reader;
        sumTf = 0;
        sumDf = numTopDocs;
        termStats = new HashMap<>();
        docTermVecs = new HashMap<>();
        this.numTopDocs = numTopDocs;
    }
    
    public IndexReader getReader() { return reader; }
    
    public Map<String, RetrievedDocTermInfo> getTermStats() {
        return termStats;
    }
    
    public void buildAllStats() throws Exception {
        int rank = 0;
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            int docId = scoreDoc.doc;
            docTermVecs.put(docId, buildStatsForSingleDoc(docId, rank, scoreDoc.score));
            rank++;
        }
    }
    
    RetrievedDocTermInfo getTermStats(String qTerm) {
        return this.termStats.get(qTerm);        
    }

    public PerDocTermVector buildStatsForSingleDoc(int docId, int rank, float sim) throws IOException {
        String termText;
        BytesRef term;
        Terms tfvector;
        TermsEnum termsEnum;
        int tf;
        RetrievedDocTermInfo trmInfo;
        PerDocTermVector docTermVector = new PerDocTermVector(docId);
        docTermVector.sim = sim;  // sim value for document D_j
        
        tfvector = reader.getTermVector(docId, Constants.CONTENT_FIELD);
        if (tfvector == null || tfvector.size() == 0)
            return null;
        
        // Construct the normalized tf vector
        termsEnum = tfvector.iterator(); // access the terms for this field
        
    	while ((term = termsEnum.next()) != null) { // explore the terms for this field
            termText = term.utf8ToString();
            tf = (int)termsEnum.totalTermFreq();
            
            // per-doc
            docTermVector.perDocStats.put(termText, new RetrievedDocTermInfo(termText, tf));
            docTermVector.sum_tf += tf;
            
            if (rank >= numTopDocs) {
                continue;
            }
            
            // collection stats for top k docs
            trmInfo = termStats.get(termText);
            if (trmInfo == null) {
                trmInfo = new RetrievedDocTermInfo(termText);
                termStats.put(termText, trmInfo);
            }
            trmInfo.tf += tf;
            trmInfo.df++;
            sumTf += tf;
            sumSim += sim;
        }
        return docTermVector;
    }
}
