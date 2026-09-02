// Ghidra headless post-script: decompile all of th07.exe + inventory + xref map
// anchored on the known bullet-pool / collision addresses.
// arg0 = output directory.
// @category Touhou
import java.io.*;
import java.util.*;
import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.*;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;
import ghidra.util.task.ConsoleTaskMonitor;

public class th07_dump extends GhidraScript {

    long POOL_LO = 0x62F958L;
    long POOL_HI = 0x650000L;

    @Override
    public void run() throws Exception {
        String[] a = getScriptArgs();
        String OUT = (a.length > 0) ? a[0] : (System.getProperty("user.home") + "/th07_re");
        new File(OUT).mkdirs();

        FunctionManager fm = currentProgram.getFunctionManager();
        ReferenceManager rm = currentProgram.getReferenceManager();
        AddressFactory af = currentProgram.getAddressFactory();
        ConsoleTaskMonitor mon = new ConsoleTaskMonitor();

        // 1. function inventory
        List<Function> funcs = new ArrayList<>();
        for (Function f : fm.getFunctions(true)) funcs.add(f);
        PrintWriter inv = new PrintWriter(new FileWriter(OUT + "/functions.csv"));
        inv.println("addr,name,size,in_refs,out_calls");
        for (Function f : funcs) {
            Address ep = f.getEntryPoint();
            int nref = 0;
            for (Reference r : rm.getReferencesTo(ep)) nref++;
            int ncall = f.getCalledFunctions(mon).size();
            inv.printf("%s,%s,%d,%d,%d%n", ep, f.getName().replace(',', '_'),
                       f.getBody().getNumAddresses(), nref, ncall);
        }
        inv.close();
        println("wrote " + funcs.size() + " functions");

        // 2. functions referencing the bullet-pool region
        Map<String, List<String>> poolHits = new LinkedHashMap<>();
        Set<String> poolFnAddrs = new HashSet<>();
        Address base = af.getDefaultAddressSpace().getAddress(0x400000L);
        ReferenceIterator rit = rm.getReferenceIterator(base);
        while (rit.hasNext()) {
            Reference ref = rit.next();
            long to = ref.getToAddress().getOffset();
            if (to < POOL_LO || to >= POOL_HI) continue;
            Address from = ref.getFromAddress();
            Function f = fm.getFunctionContaining(from);
            String key = (f != null)
                    ? (f.getEntryPoint() + "  " + f.getName()) : "(no func)";
            if (f != null) poolFnAddrs.add(f.getEntryPoint().toString());
            poolHits.computeIfAbsent(key, k -> new ArrayList<>())
                    .add(from + " -> " + ref.getToAddress() + "  "
                         + ref.getReferenceType());
        }
        PrintWriter pr = new PrintWriter(new FileWriter(OUT + "/bullet_pool_refs.txt"));
        poolHits.entrySet().stream()
                .sorted((x, y) -> y.getValue().size() - x.getValue().size())
                .forEach(e -> {
                    pr.println("=== " + e.getKey() + "   (" + e.getValue().size() + " refs)");
                    e.getValue().stream().limit(60).forEach(s -> pr.println("    " + s));
                });
        pr.close();
        println("bullet-pool region touched by " + poolHits.size() + " functions");

        // 3. decompile everything
        DecompInterface di = new DecompInterface();
        di.setOptions(new DecompileOptions());
        di.openProgram(currentProgram);
        PrintWriter big = new PrintWriter(new BufferedWriter(new FileWriter(OUT + "/decomp_all.c")));
        PrintWriter poolC = new PrintWriter(new BufferedWriter(new FileWriter(OUT + "/decomp_bullet_pool.c")));
        int done = 0;
        for (Function f : funcs) {
            String hdr = String.format("%n/* ==== %s  %s  (size %d) ==== */%n",
                    f.getEntryPoint(), f.getName(), f.getBody().getNumAddresses());
            String code;
            try {
                DecompileResults res = di.decompileFunction(f, 60, mon);
                code = (res != null && res.decompileCompleted())
                        ? res.getDecompiledFunction().getC() : "// decompile failed\n";
            } catch (Exception ex) {
                code = "// exception: " + ex + "\n";
            }
            big.print(hdr); big.print(code);
            if (poolFnAddrs.contains(f.getEntryPoint().toString())) {
                poolC.print(hdr); poolC.print(code);
            }
            if (++done % 200 == 0) println("  decompiled " + done + "/" + funcs.size());
        }
        big.close(); poolC.close();
        println("decompiled " + done + " functions -> " + OUT);
    }
}
