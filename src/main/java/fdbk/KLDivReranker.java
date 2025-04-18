package fdbk;

import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import java.util.Arrays;
import java.util.Comparator;

public class KLDivReranker implements PostFdbkReranker {
    TopDocs topDocs;
    RetrievedDocsTermStats retrievedDocsTermStats;

    public void setStats(TopDocs topDocs, RetrievedDocsTermStats retrievedDocsTermStats) {
        this.topDocs = topDocs;
        this.retrievedDocsTermStats = retrievedDocsTermStats;
    }

    @Override
    public TopDocs rerankDocs() {
        ScoreDoc[] klDivScoreDocs = new ScoreDoc[this.topDocs.scoreDocs.length];
        float klDiv;
        float p_w_D;    // P(w|D) for this doc D

        // For each document
        for (int i = 0; i < topDocs.scoreDocs.length; i++) {
            klDiv = 0;
            klDivScoreDocs[i] = new ScoreDoc(topDocs.scoreDocs[i].doc, klDiv);
            PerDocTermVector docVector = this.retrievedDocsTermStats.docTermVecs.get(topDocs.scoreDocs[i].doc);

            // For each v \in V (vocab of top ranked documents)
            for (RetrievedDocTermInfo w: retrievedDocsTermStats.termStats.values()) {
                float ntf = docVector.getNormalizedTf(w.getTerm());
                if (ntf == 0)
                    continue;
                p_w_D = ntf;
                klDiv += w.wt * Math.log(w.wt / p_w_D);
            }
            klDivScoreDocs[i].score = 1 - (float)Math.exp(klDiv);
        }
        // Sort the scoredocs in ascending order of the KL-Div scores
        klDivScoreDocs =
            Arrays.stream(klDivScoreDocs)
            .sorted((x, y) -> Float.compare(y.score, x.score)).toArray(ScoreDoc[]::new);

        TopDocs rerankedDocs = new TopDocs(topDocs.totalHits, klDivScoreDocs);
        return rerankedDocs;
    }
}
