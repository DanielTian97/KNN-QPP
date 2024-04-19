package retrieval;

import correlation.OverlapStats;
import fdbk.RelevanceModelConditional;
import fdbk.RelevanceModelIId;
import indexing.MsMarcoIndexer;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.store.FSDirectory;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import qrels.PerQueryRelDocs;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

class TermWt implements Comparable<TermWt> {
    String term;
    double wt;

    TermWt(String term, double wt) {
        this.term = term;
        this.wt = wt;
    }

    @Override
    public int compareTo(TermWt o) {
        return Double.compare(this.wt, o.wt);
    }
}

public class KNNRelModel extends SupervisedRLM {
    IndexReader qIndexReader;
    IndexSearcher qIndexSearcher;
    Map<String, MsMarcoQuery> queryMap;
    Map<String, List<MsMarcoQuery>> knnQueryMap;


    static Analyzer analyzer = MsMarcoIndexer.constructAnalyzer();

    public IndexSearcher getQueryIndexSearcher() { return qIndexSearcher; }

    public Map<String, List<MsMarcoQuery>> getKnnQueryMap() { return knnQueryMap; }

    private void constructQueriesAndQrels(String queryFile) throws Exception {
        queryMap = new HashMap<>();
        qIndexReader = DirectoryReader.open(FSDirectory.open(new File(Constants.MSMARCO_QUERY_INDEX).toPath()));
        qIndexSearcher = new IndexSearcher(qIndexReader);
        qIndexSearcher.setSimilarity(new LMDirichletSimilarity(Constants.MU));
        queryMap = constructQueries(queryFile);
    }

    public KNNRelModel(String qrelFile, String queryFile) throws Exception {
        this(qrelFile, queryFile, false, false);
    }

    public KNNRelModel(String qrelFile, String queryFile, boolean useRBO, boolean extendQV) throws Exception {
        super(qrelFile, queryFile);
        constructQueriesAndQrels(queryFile);
        constructKNNMap(useRBO, extendQV);
    }

    public KNNRelModel(String qrelFile, String queryFile, String variantsFile, boolean useRBO) throws Exception {
        super(qrelFile, queryFile);
        constructQueriesAndQrels(queryFile);

        constructKNNMap(variantsFile, useRBO);
    }

    public KNNRelModel(String qrelFile, String queryFile, String variantsFile, String variantsQidFile, String scoreFile, boolean extendToRelQueryFromDocs, boolean useRBO) throws Exception {
        super(qrelFile, queryFile);
        constructQueriesAndQrels(queryFile);

        constructKNNMap(variantsFile, variantsQidFile, scoreFile, extendToRelQueryFromDocs, useRBO);
    }

    public List<MsMarcoQuery> getQueries() { return queryMap.values().stream().collect(Collectors.toList()); }

    public Map<String, MsMarcoQuery> constructQueries(String queryFile) throws Exception {
        Map<String, String> testQueries =
                FileUtils.readLines(new File(queryFile), StandardCharsets.UTF_8)
                        .stream()
                        .map(x -> x.split("\t"))
                        .collect(Collectors.toMap(x -> x[0], x -> x[1])
                        )
                ;

        List<MsMarcoQuery> queries = new ArrayList<>();
        for (Map.Entry<String, String> e : testQueries.entrySet()) {
            String qid = e.getKey();
            String queryText = e.getValue();
            MsMarcoQuery msMarcoQuery = new MsMarcoQuery(qid, queryText, makeQuery(queryText));
            queryMap.put(qid, msMarcoQuery);
        }
        return queryMap;
    }

    void constructKNNMap(boolean useRBO, boolean extendQV) throws Exception {
        knnQueryMap = new HashMap<>();
        List<MsMarcoQuery> queries = queryMap.values().stream().collect(Collectors.toList());

        for (MsMarcoQuery q : queries) {
            List<MsMarcoQuery> knnQueries = knnQueryMap.get(q.getId());
            if (knnQueries == null) {
                // knnQueries = q.retrieveSimilarQueries(
                //         getQueryIndexSearcher(),
                //         Constants.QPP_COREL_MAX_VARIANTS)
                // ;

                // // Replace BM25 similarities with RBO similarities. Just to be consistent with gen variants...
                // if (useRBO) {
                //     for (MsMarcoQuery knnQuery : knnQueries)
                //         knnQuery.setRefSim(computeRBO(q, knnQuery));

                //     knnQueries = knnQueries.stream().sorted(Comparator.reverseOrder()).collect(Collectors.toList());
                //     //knnQueries.stream().forEach(System.out::println);
                // }
                knnQueries = retrieveBM25KnnQueries(q, useRBO);
                

                if(extendQV) {
                    List<MsMarcoQuery> qVExtensions = extendRetrievedKnnQueries(knnQueries);
                    knnQueries.addAll(qVExtensions);

                    // rerank
                    for (MsMarcoQuery knnQuery : knnQueries) {
                        knnQuery.setRefSim(computeRBO(q, knnQuery));
                    }
                    // sort after reranking
                    knnQueries = knnQueries.stream().sorted(Comparator.reverseOrder()).collect(Collectors.toList());
                }

                knnQueryMap.put(q.getId(), knnQueries);
                
                for (MsMarcoQuery rq : knnQueryMap.get(q.getId())) {
                    System.out.println(rq);
                }
            }
        }
    }

    // void constructKNNMapFlexibly(boolean useRetrievedQV, boolean extendRetrievedQV, boolean useGeneratedQV) throws Exception {
        
    // }

    List<MsMarcoQuery> retrieveBM25KnnQueries (MsMarcoQuery q, boolean useRBO) throws Exception {
        List<MsMarcoQuery> knnQueries = q.retrieveSimilarQueries(this.rels, getQueryIndexSearcher(), Constants.QPP_COREL_MAX_VARIANTS);

        // Replace BM25 similarities with RBO similarities. Just to be consistent with gen variants...
        if (useRBO) {
            for (MsMarcoQuery knnQuery : knnQueries) {
                knnQuery.setRefSim(computeRBO(q, knnQuery));
            }

            knnQueries = knnQueries.stream().sorted(Comparator.reverseOrder()).collect(Collectors.toList());
        }

        // //for analysis the weight
        // for(MsMarcoQuery aq : knnQueries){
        //     System.out.printf("%f ", aq.getRefSim());
        // }
        // System.out.print('\n');

        return knnQueries;
    }

    List<MsMarcoQuery> extendRetrievedKnnQueries (List<MsMarcoQuery> RetrievedKnnQueries) throws Exception {
        List<MsMarcoQuery> qVExtensions = new ArrayList<MsMarcoQuery>();

        // System.out.print("rqs: ");
        // for (MsMarcoQuery rq : RetrievedKnnQueries) {
        //     System.out.print(rq.getId());
        //     System.out.print(" ");
        // }
        // System.out.print("\n");
        
        for (MsMarcoQuery rq : RetrievedKnnQueries) {
            PerQueryRelDocs relDocs = rq.getRelDocSet();
            if (relDocs==null || relDocs.getRelDocs().isEmpty())
                continue;
            String docName = relDocs.getRelDocs().iterator().next();
            String docText = reader.document(this.getDocOffset(docName)).get(Constants.CONTENT_FIELD);
            MsMarcoQuery docQuery = new MsMarcoQuery(docName, docText);

            List<MsMarcoQuery> foundQueriesForQ;
            if(Constants.ABLATION_EXP_FOR_EXTENDING){
                foundQueriesForQ = rq.retrieveSimilarQueries(getQueryIndexSearcher(), Constants.CLARITY_CAL_RANGE);
            } else {
                foundQueriesForQ = docQuery.retrieveSimilarQueries(getQueryIndexSearcher(), Constants.CLARITY_CAL_RANGE);
            }
            
            // System.out.print("\nfor rq: ");
            // System.out.print(rq.getId());
            // System.out.print(" retrieved: ");
            // for(MsMarcoQuery rrq : foundQueriesForQ){
            //     System.out.print(rrq.getId());
            //     System.out.print(" ");
            // }
            // System.out.print("\n");

            for(MsMarcoQuery rrq : foundQueriesForQ){
                boolean kept = true;

                // System.out.println("\nCheck QVs:");
                for(MsMarcoQuery rqC : RetrievedKnnQueries){
                    // System.out.printf("%s %s %b", rqC.getId(), rrq.getId(), rqC.getId() == rrq.getId());
                    if(rrq.getId().equals(rqC.getId())){ //rqC means rq for comparison
                        // foundQueriesForQ.remove(rrq);
                        kept = false;
                        break;
                    }
                }

                // System.out.println("\nCheck QVEs:");
                if(kept) {
                    for (MsMarcoQuery qVE : qVExtensions) {
                        // System.out.printf("%s %s %b", qVE.getId(), rrq.getId(), qVE.getId() == rrq.getId());
                        if(qVE.getId().equals(rrq.getId())){
                            kept = false;
                            break;
                        }
                    }
                    if(kept){
                        //System.out.println(rq.getId());
                        //System.out.println(rq.getQuery());
                        qVExtensions.add(rrq);
                    }
                }
            }

        }
        
        // System.out.print("\ne qvs: ");
        // for (MsMarcoQuery eqv : qVExtensions) {
        //     System.out.print(eqv.getId());
        //     System.out.print(" ");
        // }
        // System.out.println("\n");

        return qVExtensions;
    }

    float computeRBO(MsMarcoQuery q, MsMarcoQuery refQ) {
        TopDocs topA = null, topB = null;
        try {
            topA = searcher.search(q.getQuery(), Constants.RBO_NUM_DOCS);
            topB = searcher.search(refQ.getQuery(), Constants.RBO_NUM_DOCS);
        }
        catch (Exception ex) { ex.printStackTrace(); }

        return topA==null||topB==null? 0 : (float)OverlapStats.computeRBO(topA, topB);
    }

    void constructKNNMap(String variantsFile, String variantsQidFile, String scoreFile, boolean extendQV, boolean useRBO) throws Exception {
        knnQueryMap = new HashMap<>();

        List<String> textLines = FileUtils.readLines(new File(variantsFile), StandardCharsets.UTF_8);
        List<String> qidLines = FileUtils.readLines(new File(variantsQidFile), StandardCharsets.UTF_8);
        List<String> scoreLines = FileUtils.readLines(new File(scoreFile), StandardCharsets.UTF_8);

        Iterator<String> textLineIterator = textLines.iterator();
        Iterator<String> qidLineIterator = qidLines.iterator();
        Iterator<String> scoreLineIterator = scoreLines.iterator();

        while(textLineIterator.hasNext()){
            String textLine = textLineIterator.next();
            String qidLine = qidLineIterator.next();
            String scoreLine = scoreLineIterator.next();

            String[] tokens = textLine.split("\\t");
            String[] qids = qidLine.split("\\t");
            String[] scores = scoreLine.split("\\t");

            String qid = tokens[0];
            MsMarcoQuery q = queryMap.get(qid);
            
            List<MsMarcoQuery> knnQueries = knnQueryMap.get(qid);
            // for (int i=2; i < tokens.length; i++) {
            for (int i=2; i < 2+Constants.QPP_COREL_MAX_VARIANTS; i++) {
                // List<MsMarcoQuery> knnQueries = knnQueryMap.get(qid);
                if (knnQueries==null) {
                    knnQueries = new ArrayList<>();
                    knnQueryMap.put(qid, knnQueries);
                }
                MsMarcoQuery rq = new MsMarcoQuery(qids[i], tokens[i], Float.parseFloat(scores[i]));
                if(rq.query==null){
                    continue;
                }
                rq.relDocs = rels.getRelInfo(rq.qid);

                // // aborted!!!!!
                // if (extendToRelQueryFromDocs){
                //     // System.out.println(knnQueries.size());
                //     // queries which are in fact docs
                //     knnQueries = getRelDocsSimQuery(rq, knnQueries);
                //     // System.out.println(knnQueries.size());
                // }

                MsMarcoQuery testQuery = queryMap.get(qid);
                if (testQuery==null)
                    continue;  // the variants file is a union of dl'19 and 20... hence safe to discard missing ones
                // rq.setRefSim(1.0f); // uniform

                knnQueries.add(rq);

            }

            if (useRBO) {
                for (MsMarcoQuery knnQuery : knnQueries)
                    knnQuery.setRefSim(computeRBO(q, knnQuery));
                
                    knnQueries = knnQueries.stream().sorted(Comparator.reverseOrder()).collect(Collectors.toList());
                // knnQueries = knnQueries.stream().limit(Constants.QPP_COREL_MAX_VARIANTS).sorted(Comparator.reverseOrder()).collect(Collectors.toList());
                //knnQueries.stream().forEach(System.out::println);
            }

            // I realise it is not needed to create a special method for it
            if(extendQV) {

                // System.out.print("Before extension: ");
                // for (MsMarcoQuery knnQuery : knnQueries) {
                //     System.out.print(knnQuery.getId());
                //     System.out.print(" ");
                // }
                // System.out.println("\n");

                List<MsMarcoQuery> qVExtensions = extendRetrievedKnnQueries(knnQueries);
                knnQueries.addAll(qVExtensions);

                // rerank
                for (MsMarcoQuery knnQuery : knnQueries) {
                    knnQuery.setRefSim(computeRBO(q, knnQuery));
                }
                // sort after reranking
                knnQueries = knnQueries.stream().sorted(Comparator.reverseOrder()).limit(Constants.QPP_COREL_MAX_VARIANTS).collect(Collectors.toList());

                // System.out.print("\nAfter extension: ");
                // for (MsMarcoQuery knnQuery : knnQueries) {
                //     System.out.print(knnQuery.getId());
                //     System.out.print(" ");
                // }
                // System.out.println("\n");
                
            }

            knnQueryMap.put(qid, knnQueries);

            for (MsMarcoQuery rq : knnQueryMap.get(q.getId())) {
                System.out.println(rq);
            }

            // //for analysis the weight
            // for(MsMarcoQuery aq : knnQueries){
            //     System.out.printf("%f ", aq.getRefSim());
            // }
            // System.out.print('\n');
        }     
    }


    void constructKNNMap(String variantsFile, boolean useRBO) throws Exception {
        knnQueryMap = new HashMap<>();

        List<String> lines = FileUtils.readLines(new File(variantsFile), StandardCharsets.UTF_8);

        for (String line: lines) {
            String[] tokens = line.split("\\t");
            String qid = tokens[0];
            MsMarcoQuery q = queryMap.get(qid);

            List<MsMarcoQuery> knnQueries = knnQueryMap.get(qid);
            for (int i=2; i < tokens.length; i++) {
                // List<MsMarcoQuery> knnQueries = knnQueryMap.get(qid);
                if (knnQueries==null) {
                    knnQueries = new ArrayList<>();
                    knnQueryMap.put(qid, knnQueries);
                }
                MsMarcoQuery rq = new MsMarcoQuery(qid + "_v_" + i, tokens[i]);
                if(rq.query==null){
                    continue;
                }

                MsMarcoQuery testQuery = queryMap.get(qid);
                if (testQuery==null)
                    continue;  // the variants file is a union of dl'19 and 20... hence safe to discard missing ones
                rq.setRefSim(1.0f); // uniform

                knnQueries.add(rq);
            }

            if (useRBO) {
                for (MsMarcoQuery knnQuery : knnQueries)
                    knnQuery.setRefSim(computeRBO(q, knnQuery));

                knnQueries = knnQueries.stream().sorted(Comparator.reverseOrder()).collect(Collectors.toList());
                //knnQueries.stream().forEach(System.out::println);
            }

            knnQueryMap.put(qid, knnQueries);
            //List<MsMarcoQuery> knnQueries = knnQueryMap.get(qid);
            //knnQueries.stream().forEach(System.out::println);

            // //for analysis the weight
            // for(MsMarcoQuery aq : knnQueries){
            //     System.out.printf("%f ", aq.getRefSim());
            // }
            // System.out.print('\n');
            for (MsMarcoQuery rq : knnQueryMap.get(q.getId())) {
                System.out.println(rq);
            }

        }
    }

    public List<MsMarcoQuery> getKNNs(MsMarcoQuery q, int k) {
        List<MsMarcoQuery> knnQueries = knnQueryMap.get(q.getId());
        return knnQueries.subList(0, Math.min(k, knnQueries.size()));
    }

    int findRank(String docId, TopDocs topDocs) throws Exception {
        int key = getDocOffset(docId);
        int rank = topDocs.scoreDocs.length;
        for (int i=0; i < topDocs.scoreDocs.length; i++) {
            ScoreDoc sd = topDocs.scoreDocs[i];
            if (key == sd.doc) {
                rank = i;
                break;
            }
        }
        return rank+1;
    }

    private List<MsMarcoQuery> getRelDocsSimQuery(MsMarcoQuery q, List<MsMarcoQuery> knnQueries) throws Exception {
        Set<String> relDocs = q.getRelDocSet().getRelDocs();
        for (String docName : relDocs) {
            int docId = getDocOffset(docName);
            String docText = searcher.getIndexReader().document(docId).get(Constants.CONTENT_FIELD);
            // System.out.println(docText);
            // append this to the variants file
            MsMarcoQuery docQuery = new MsMarcoQuery(docName, docText, q.getRefSim());
            knnQueries.add(docQuery);
        }
        return knnQueries;
    }

    List<MsMarcoQuery> genFewShotExamples(MsMarcoQuery query, int k) {
        query.fewshotInfo = new JSONObject();
        query.fewshotInfo.put("trecdl.query.id", query.qid);
        query.fewshotInfo.put("trecdl.query.text", query.qText);
        JSONArray relatedQueries = new JSONArray();

        try {
            Query luceneQuery = makeQuery(query.qText);
            List<MsMarcoQuery> knnQueries = new ArrayList<>();
            Map<String, Double> rel;

            TopDocs knnQueriesTopDocs = qIndexSearcher.search(luceneQuery, k);
            for (ScoreDoc sd : knnQueriesTopDocs.scoreDocs) {
                Document q = qIndexReader.document(sd.doc);
                knnQueries.add(
                        new MsMarcoQuery(
                            q.get(Constants.ID_FIELD),
                            q.get(Constants.CONTENT_FIELD),
                            //MsMarcoIndexer.analyze(analyzer, q.get(Constants.CONTENT_FIELD)),
                            sd.score)
                );
            }

            knnQueries.stream().forEach(x -> x.simWithOrig = computeRBO(query, x));
            knnQueries = knnQueries.stream().sorted(Comparator.reverseOrder()).collect(Collectors.toList());
            //knnQueries.forEach(System.out::println);

            int relQIndex, relDocIndex;

            System.out.println(String.format("Query: %s", query.qText));
            relQIndex = 0;

            for (MsMarcoQuery rq: knnQueries) {
                relQIndex++;
                //System.out.println(String.format("Related Query %d: %s", relQIndex, rq.qText));
                JSONObject rq_json = new JSONObject();
                rq_json.put("msmarco.query.id", rq.qid);
                rq_json.put("msmarco.query.text", rq.qText);
                rq_json.put("msmarco.query.rank", relQIndex);
                JSONArray relDocsJsonArray = new JSONArray();

                rq.query = makeQuery(rq.qText);
                System.out.println("Executing top-100 on related query " + rq.query);
                TopDocs topDocsRQ = searcher.search(rq.query, 1000);
                // Find the ranks of the rel and the nonrel docs

                PerQueryRelDocs relDocIds = rels.getRelInfo(rq.qid);
                if (relDocIds == null) continue;

                relDocIndex = 0;
                JSONObject docInfoJsonObj = new JSONObject();
                for (String docId : relDocIds.getRelDocs()) {
                    relDocIndex++;

                    String relDocText = reader.document(getDocOffset(docId)).get(Constants.CONTENT_FIELD);
                    System.out.println(String.format("Query %d [%s], RelDoc %d [%s]  %s", relQIndex, rq.qid, relDocIndex, docId, relDocText));
                    System.out.println("Reldoc rank: " + findRank(docId, topDocsRQ));

                    docInfoJsonObj.put("reldoc.id", docId);
                    docInfoJsonObj.put("reldoc.text", relDocText);
                    docInfoJsonObj.put("reldoc.lexmodel.rank", findRank(docId, topDocsRQ));

                    // We also need to provide one non-rel doc. For that we execute a LM-Dir/BM25
                    // on this query and sample a doc at random from ranks 50 to 100.
                    int sampled_negative_index = 50 + (int)(Math.random() * 100);

                    Document negDoc = reader.document(topDocsRQ.scoreDocs[sampled_negative_index].doc);
                    String nonRelDocId = negDoc.get(Constants.ID_FIELD);
                    String nonRelDocText = negDoc.get(Constants.CONTENT_FIELD);
                    System.out.println(String.format("Query %d [%s], NonRelDoc %d [%s]: %s", relQIndex, rq.qid, relDocIndex, nonRelDocId, nonRelDocText));
                    System.out.println("NonReldoc rank: " + sampled_negative_index);

                    docInfoJsonObj.put("nreldoc.id", nonRelDocId);
                    docInfoJsonObj.put("nreldoc.text", nonRelDocText);
                    docInfoJsonObj.put("nreldoc.lexmodel.rank", sampled_negative_index);

                }
                relDocsJsonArray.add(docInfoJsonObj);
                rq_json.put("msmarco.qrel.info", relDocsJsonArray);
                relatedQueries.add(rq_json);
            }
            query.fewshotInfo.put("fewshots", relatedQueries);

            //System.out.println(knnQueries.stream().map(x -> x.qText).collect(Collectors.joining("|")));
            return knnQueries;
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }
    }

    Query rocchioQE(MsMarcoQuery query, TopDocs topDocs) throws Exception {
        // original query terms
        BooleanQuery.Builder qb = new BooleanQuery.Builder();
        Set<String> queryTerms = query.getQueryTermsAsString();

        Map<String, Double> rel, nonrel;
        Map<String, Double> origQ = queryTerms.stream().collect(
                Collectors.toMap(x -> x, x -> Constants.ROCCHIO_ALPHA)); // orig query terms
        Map<String, Double> relAcc = new HashMap<>(), nonRelAcc = new HashMap<>();

        try {
            List<MsMarcoQuery> knnQueries = genFewShotExamples(query, Constants.K);

            for (MsMarcoQuery knnQ: knnQueries) {
                PerQueryRelDocs relDocIds = rels.getRelInfo(knnQ.qid);
                if (relDocIds == null) continue;

                for (String docId: relDocIds.getRelDocs()) {
                    rel = makeLMTermWts(docId);
                    mergeInto(rel, relAcc); // sum <- sum + beta*rel (rel and sum are vecs, beta is scalar)
                }

                List<Integer> nonRelDocIds = new ArrayList<>();
                for (ScoreDoc sd: topDocs.scoreDocs) {
                    if (!relDocIds.getRelDocs().contains(reader.document(sd.doc).get(Constants.ID_FIELD))) {
                        nonRelDocIds.add(sd.doc);
                        if (nonRelDocIds.size() >= Constants.ROCCHIO_NUM_NEGATIVE)
                            break;
                    }
                }
                for (Integer docId: nonRelDocIds) {
                    nonrel = makeLMTermWts(docId);
                    mergeInto(nonrel, nonRelAcc); // sum <- sum + beta*rel (rel and sum are vecs, beta is scalar)
                }

                mergeInto(relAcc, Constants.ROCCHIO_BETA/relDocIds.getRelDocs().size(), origQ);
                mergeInto(nonRelAcc, -1*Constants.ROCCHIO_GAMMA/nonRelDocIds.size(), origQ);
            }
            origQ = origQ.entrySet()
                .stream()
                .filter(e-> e.getValue() > 0)
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .limit(Constants.NUM_EXPANSION_TERMS)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2, LinkedHashMap::new)
            );
        }
        catch (Exception ex) { ex.printStackTrace(); }

        origQ
        .entrySet().stream()
        .forEach(e ->
            qb.add(
                new BooleanClause(
                new BoostQuery(
                    new TermQuery(new Term(Constants.CONTENT_FIELD, e.getKey())),
                    (float)e.getValue().doubleValue()
                ),
                BooleanClause.Occur.SHOULD)
            )
        );

        Query expandedQuery = qb.build();
        return expandedQuery;
    }

    BooleanQuery makeQueryWithExpansionTerms(MsMarcoQuery query) {
        String qid = query.qid;
        String queryText = query.qText;

        // original query terms
        BooleanQuery.Builder qb = new BooleanQuery.Builder();
        Set<String> queryTerms =
            Arrays.stream(MsMarcoIndexer
                .analyze(MsMarcoIndexer.constructAnalyzer(), queryText)
                .split("\\s+"))
                .collect(Collectors.toSet())
        ;

        try {
            termDistributions.clear();
            List<MsMarcoQuery> knnQueries = genFewShotExamples(query, Constants.K);
            for (MsMarcoQuery knnQ: knnQueries) {
                fit(knnQ.qid, knnQ.qText);
            }
        }
        catch (Exception ex) { ex.printStackTrace(); }

        queryTerms.stream()
            .map(x-> new BoostQuery(new TermQuery(new Term(Constants.CONTENT_FIELD, x)), 1.0f))
            .forEach(tq -> qb.add(new BooleanClause(tq, BooleanClause.Occur.SHOULD)));
        ;

        termDistributions
            .values()
            .stream()
            .flatMap(x->
                x.cooccurProbs
                .entrySet()
                .stream()
                .map(e -> new TermWt(e.getKey(), e.getValue()))
            )
            .sorted(Comparator.reverseOrder())
            .limit(Constants.NUM_EXPANSION_TERMS)
            .forEach(termWt ->
                qb.add(new BooleanClause(
                new BoostQuery(
                    new TermQuery(new Term(Constants.CONTENT_FIELD, termWt.term)),
                    (float)termWt.wt
                ),
                BooleanClause.Occur.SHOULD))
        );

        return qb.build();
    }

    void findKNNOfQueries(String trecDLQueryFile, String outJSONFile) throws Exception {
        Map<String, String> testQueries =
        loadQueries(trecDLQueryFile)
        .entrySet()
        .stream()
        .collect(
            Collectors.toMap(
                e -> e.getKey(),
                e -> MsMarcoIndexer.normalizeNumbers(e.getValue()
                )
            )
        );
        System.out.println(testQueries);

        BufferedWriter bw = new BufferedWriter(new FileWriter(outJSONFile));
        bw.write("[");
        for (Map.Entry<String, String> e: testQueries.entrySet()) {
            MsMarcoQuery testQuery = new MsMarcoQuery(e.getKey(), e.getValue());
            System.out.println(testQuery);
            genFewShotExamples(testQuery, Constants.K);
            bw.write(testQuery.fewshotInfo.toJSONString());
            bw.write(",");
            bw.newLine();
        }
        bw.write("]");
        bw.close();
    }

    public void retrieve() throws Exception {
        Map<String, String> testQueries = loadQueries(Constants.QUERY_FILE_TEST);
        testQueries
                .entrySet()
                .stream()
                .collect(
                    Collectors.toMap(
                        e -> e.getKey(),
                        e -> MsMarcoIndexer.normalizeNumbers(e.getValue())
                    )
                )
        ;

        TopDocs topDocs = null;
        Map<String, TopDocs> topDocsMap = new HashMap<>(queries.size());

        for (Map.Entry<String, String> e : testQueries.entrySet()) {
            MsMarcoQuery query = new MsMarcoQuery(e.getKey(), e.getValue());
            Query luceneQuery = query.getQuery();
            topDocs = searcher.search(luceneQuery, Constants.NUM_WANTED); // descending BM25
            topDocsMap.put(query.qid, topDocs);

            //Query luceneQuery = Constants.QRYEXPANSION? makeQueryWithExpansionTerms(query) : makeQuery(query);
        }

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(Constants.RES_FILE))) {
            for (Map.Entry<String, String> e : testQueries.entrySet()) {
                MsMarcoQuery query = new MsMarcoQuery(e.getKey(), e.getValue());
                String qid = query.qid;
                String queryText = query.qText;

                topDocs = topDocsMap.get(qid);
                if (Constants.RLM)
                    topDocs = rlm(searcher, query, topDocs);
                else if (Constants.RERANK)
                    topDocs = rerank(query, topDocs);
                else if (Constants.QRYEXPANSION) {
                    Query expandedQuery = rocchioQE(query, topDocs);
                    System.out.println("Expanded query: " + expandedQuery.toString());
                    topDocs = searcher.search(expandedQuery, Constants.NUM_WANTED);
                    topDocs = rlm(searcher, new MsMarcoQuery(searcher, qid, expandedQuery), topDocs);
                }

                AtomicInteger rank = new AtomicInteger(1);
                int rel = 0;
                for (ScoreDoc sd : topDocs.scoreDocs) {
                    String docName = reader.document(sd.doc).get(Constants.ID_FIELD);
                    PerQueryRelDocs perQueryRelDocs = eval_rels.getRelInfo(qid);
                    rel = perQueryRelDocs.isRel(docName)? 1 : 0;

                    bw.write(String.format(
                            "%s\tQ0\t%s\t%d\t%.6f\t%d\t%s",
                            qid, docName,
                            rank.getAndIncrement(), sd.score,
                            rel,
                            reader.document(sd.doc).get(Constants.CONTENT_FIELD)
                    ));
                    bw.newLine();
                }
            }
        }
    }

    static void mergeInto(Map<String, Double> a, Map<String, Double> b) { // merge a into b
        mergeInto(a, 1, b);
    }

    static void mergeInto(Map<String, Double> a, double weight_a, Map<String, Double> b) { // merge a into b
        for (Map.Entry<String, Double> e: a.entrySet()) {
            String key = e.getKey();
            Double a_val = e.getValue() * weight_a;
            Double b_val = b.get(key);
            if (b_val == null) {
                b_val = 0.0;
            }
            b_val += a_val;
            b.put(key, b_val);
        }
    }

    public Map<String, Double> makeAvgLMDocModel(List<MsMarcoQuery> queries) throws Exception {
        Map<String, Double> docModel, avgDocModel = new HashMap<>();

        for (MsMarcoQuery query: queries) {
            PerQueryRelDocs relDocIds = query.relDocs;
            if (relDocIds == null)
                relDocIds = rels.getRelInfo(query.qid);
            if (relDocIds == null)
                continue;

            for (String docId : relDocIds.getRelDocs()) {
                docModel = makeLMTermWts(docId);
                mergeInto(docModel, query.simWithOrig, avgDocModel);
            }
        }

        double l2Norm = TermDistribution.l2Norm(avgDocModel);
        return avgDocModel.entrySet().stream().collect(Collectors.toMap(e->e.getKey(), e->e.getValue()/l2Norm));
    }

    // Use the supervised RLM to rerank results
    TopDocs rerank(MsMarcoQuery query, TopDocs retrievedRes) throws Exception {
        String queryText = query.qText;
        ScoreDoc[] rerankedScoreDocs = new ScoreDoc[retrievedRes.scoreDocs.length];
        Map<String, Double> knnDocTermWts, thisDocTermWts;
        int i = 0;
        double p_R_d = 0;

        List<MsMarcoQuery> knnQueries = genFewShotExamples(query, Constants.K);

        knnDocTermWts = makeAvgLMDocModel(knnQueries); // make a centroid of reldocs for this query
        if (knnDocTermWts == null)
            return retrievedRes;

        for (ScoreDoc sd: retrievedRes.scoreDocs) {
            rerankedScoreDocs[i] = new ScoreDoc(sd.doc, sd.score);
            // LM model for this doc
            thisDocTermWts = makeLMTermWts(sd.doc, true);

            p_R_d = TermDistribution.cosineSim(knnDocTermWts, thisDocTermWts);
            rerankedScoreDocs[i++].score = (float)p_R_d * sd.score;
        }

        rerankedScoreDocs = Arrays.stream(rerankedScoreDocs)
                .sorted((o1, o2) -> o1.score < o2.score? 1: o1.score==o2.score? 0 : -1) // sort descending by sims
                .collect(Collectors.toList())
                .toArray(rerankedScoreDocs)
        ;

        return new TopDocs(new TotalHits(rerankedScoreDocs.length, TotalHits.Relation.EQUAL_TO), rerankedScoreDocs);
    }

    TopDocs srlm(IndexSearcher searcher, MsMarcoQuery query, TopDocs topDocs) throws Exception {
        List<MsMarcoQuery> knnQueries = genFewShotExamples(query, Constants.K);
        List<ScoreDoc> relDocs = new ArrayList<>();
        for (MsMarcoQuery knnQuery: knnQueries) {
            PerQueryRelDocs relDocIds = rels.getRelInfo(knnQuery.qid);
            if (relDocIds == null)
                continue;

            // add the rels for similar queries
            for (String docId : relDocIds.getRelDocs()) {
                relDocs.add(new ScoreDoc(getDocOffset(docId), 1.0f));
            }
        }
        float sum_scores = relDocs.stream().map(x->x.score).reduce(0.0f, (a, b) -> a+b);
        relDocs.stream().forEach(x -> x.score/= sum_scores);

        float sum_scores_topdocs =
                Arrays.stream(topDocs.scoreDocs)
                .map(x->x.score).reduce(0.0f, (a, b) -> a+b);
        for (ScoreDoc sd: topDocs.scoreDocs) { // add the retrieved ones
            relDocs.add(new ScoreDoc(sd.doc, sd.score/sum_scores_topdocs));
        }

        TopDocs relTopDocs = new TopDocs(
            new TotalHits(relDocs.size(), TotalHits.Relation.EQUAL_TO),
            relDocs.stream().toArray(ScoreDoc[]::new)
        );

        RelevanceModelIId fdbkModel = new RelevanceModelConditional(
                searcher, query, relTopDocs, Constants.RLM_NUM_TOP_DOCS);
        fdbkModel.computeFdbkWeights();
        return fdbkModel.rerankDocs(topDocs);
    }

    TopDocs rlm(IndexSearcher searcher, MsMarcoQuery query, TopDocs topDocs) throws Exception {
        RelevanceModelIId fdbkModel = new RelevanceModelConditional(
                searcher, query, topDocs, Constants.RLM_NUM_TOP_DOCS);
        fdbkModel.computeFdbkWeights();
        TopDocs reranked = fdbkModel.rerankDocs();
        if (!Constants.RLM_POST_QE)
            return reranked;

        MsMarcoQuery expanded_query = fdbkModel.expandQuery(Constants.NUM_EXPANSION_TERMS);
        return searcher.search(expanded_query.query, Constants.NUM_WANTED);
    }

    public static void main(String[] args) {
        try {
            System.out.println("Loading queries");
            KNNRelModel knnRelModel = new KNNRelModel(Constants.QRELS_TRAIN, "data/trecdl/trecdl1920.txt");
            //knnRelModel.retrieve();

            if (args.length<2) {
                System.out.println("usage: retrieval.KNNRelModel <TREC DL evaluation query file (2019/2020)> <json file>");
                args = new String[2];
                args[0] = "data/trecdl/trecdl1920.txt";
                args[1] = Constants.FEWSHOT_JSON;
            }

            knnRelModel.findKNNOfQueries(args[0], args[1]);
        }
        catch (Exception ex) { ex.printStackTrace(); }
    }
}
