package indexing;

import retrieval.Constants;

public class BEIRIndexer extends MsMarcoIndexer {

    public static void main(String[] args) {
        try {
            MsMarcoIndexer indexer = new MsMarcoIndexer();

            indexer.indexCollection(Constants.COVID_COLL, Constants.COVID_INDEX);
            System.out.println("Indexing queries for TREC-COVID...");

            indexer.indexCollection(Constants.FEVER_COLL, Constants.FEVER_INDEX);
            System.out.println("Indexing queries for FEVER...");

            indexer.indexCollection(Constants.WEBIS_COLL, Constants.WEBIS_INDEX);
            System.out.println("Indexing queries for WEBIS_2020...");
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

}