fileID = fopen('./incl_img_latex.txt', 'w');
fprintf(fileID, '\\begin{figure}[h!]\n\\centering\n');

files = dir('./*.png');
for file = files'
    str = file.name;
    fprintf(fileID, '\\begin{subfigure}[b]{0.2\\textwidth}\n\\includegraphics[trim=45 180 70 210,clip,width=\\textwidth]{%s}\n\\end{subfigure}\n', str);
end

fprintf(fileID, '\\caption{C.}\n\\label{fig:}\n\\end{figure}');
fclose(fileID);