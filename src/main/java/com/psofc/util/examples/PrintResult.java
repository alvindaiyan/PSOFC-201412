package com.psofc.util.examples;


import java.util.HashMap;
import java.util.Map;

/**
 * Created by daiyan on 16/12/14.
 */
public abstract class PrintResult
{
    interface Record{}

    private Map<String, Record> resultLookup = new HashMap<String, Record>();


    public void addRecord(String name, Record r)
    {
        if(resultLookup != null && !resultLookup.containsKey(name) )
        {
            resultLookup.put(name, r);
        }
        else
        {
            throw new IllegalArgumentException("error name already exists or resultLookup is null");
        }
    }

    public abstract void prettyPrint();

}
