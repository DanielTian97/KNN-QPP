package experiments;

import indexing.MsMarcoIndexer;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import qpp.*;
import qrels.AllRetrievedResults;
import qrels.RetrievedResults;
import retrieval.Constants;
import retrieval.MsMarcoQuery;
import retrieval.OneStepRetriever;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class QPPEvaluatorSimple {

    static class TauAndSARE {
        double tau;
        double sare;

        TauAndSARE(double tau, double sare) {
            this.tau = tau;
            this.sare = sare;
        }
    }

    static Query makeQuery(String queryText) throws Exception {
        BooleanQuery.Builder qb = new BooleanQuery.Builder();
        String[] tokens = MsMarcoIndexer.analyze(MsMarcoIndexer.constructAnalyzer(), queryText).split("\\s+");
        for (String token: tokens) {
            TermQuery tq = new TermQuery(new Term(Constants.CONTENT_FIELD, token));
            qb.add(new BooleanClause(tq, BooleanClause.Occur.SHOULD));
        }
        return (Query)qb.build();
    }

    static Map<String, MsMarcoQuery> constructQueries(String queryFile, Map<String, MsMarcoQuery> queryMap) throws Exception {
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

    static TauAndSARE runExperiment(
            String baseQPPModelName, // nqc/uef
            IndexSearcher searcher,
            AllRetrievedResults res,
            List<MsMarcoQuery> queries,
            Map<String, TopDocs> topDocsMap) {

        double tau = 0;
        double sare = 0;

        QPPMethod baseModel;
        if(baseQPPModelName.equals("rsd")) {
            baseModel = new RSDSpecificity(new NQCSpecificity(searcher));
        } else {
            baseModel = baseQPPModelName.equals("nqc") ? new NQCSpecificity(searcher) : new UEFSpecificity(new NQCSpecificity(searcher));
        }

        int numQueries = queries.size();
        double[] qppEstimates = new double[numQueries];
        String[] qids = new String[numQueries];

        int i = 0;       

        for (MsMarcoQuery query : queries) {

            RetrievedResults rr = res.getRetrievedResultsForQueryId(query.getId());

            TopDocs topDocs = topDocsMap.get(query.getId());

            try {
                qppEstimates[i] = (float) baseModel.computeSpecificity(
                        query, rr, topDocs, Constants.QPP_NUM_TOPK, false);
                qids[i] = query.getId();
            } catch (Exception e) {
                System.out.println(query.getId());
            }

            System.out.println(String.format("%s: QPP = %.4f", query.getId(), qppEstimates[i]));
            i++;
        }

        return new TauAndSARE(tau, sare);
    }

    static void trainAndTest(
            String baseModelName,
            OneStepRetriever retriever,
            String queryFile,
            String trainResFile
    )
            throws Exception {
        IndexSearcher searcher = retriever.getSearcher();
        AllRetrievedResults retRcds = new AllRetrievedResults(trainResFile);

        Map<String, MsMarcoQuery> queryMap = new HashMap<>();
        queryMap = constructQueries(queryFile, queryMap);
        List<MsMarcoQuery> trainQueries = queryMap.values().stream().collect(Collectors.toList());
        Map<String, TopDocs> topDocsMap = retRcds.castToTopDocs();

        TauAndSARE analyseResults = runExperiment(baseModelName,
                searcher, retRcds,
                trainQueries, topDocsMap);

    }

    public static double calculateVariation(double[] array) {

        // get the sum of array
        double sum = 0.0;
        for (double i : array) {
            sum += i;
        }
    
        // get the mean of array
        int length = array.length;
        double mean = sum / length;
    
        // calculate the standard deviation
        double standardDeviation = 0.0;
        for (double num : array) {
            standardDeviation += Math.pow(num - mean, 2);
        }
    
        return standardDeviation / length;
    }

    public static void main(String[] args) {

        if (args.length < 3) {
            System.out.println("Required arguments: <retriever> <queryset name> <uef/nqc>");
            args = new String[3];
            args[0] = "bm25";
            args[1] = "k0";
            args[2] = "nqc";
        }

        String pathFormat = "data/answer_queries/queries_%s.queries";
        String queryPath = String.format(pathFormat, args[1]);

        String runPathFormat = "specialRuns/%s.%s.res";
        String runPath = String.format(runPathFormat, args[0], args[1]);

        try {
            OneStepRetriever retriever = new OneStepRetriever(Constants.QUERY_FILE_TEST);
            Settings.init(retriever.getSearcher());

            trainAndTest(args[2], retriever,
                    queryPath,
                    runPath);
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

}
