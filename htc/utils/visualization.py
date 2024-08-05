# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import base64
import gzip
import json
import math
import re
import uuid
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, display
from matplotlib.colors import to_rgba
from PIL import Image
from plotly.subplots import make_subplots
from scipy import stats

from htc.cpp import tensor_mapping
from htc.evaluation.metrics.ECE import ECE
from htc.evaluation.metrics.scores import dice_from_cm
from htc.models.common.MetricAggregation import MetricAggregation
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.colorscale import tivita_colorscale
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings
from htc.tivita.hsi import tivita_wavelengths
from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
from htc.utils.ColorcheckerReader import ColorcheckerReader
from htc.utils.colors import generate_distinct_colors
from htc.utils.Config import Config
from htc.utils.helper_functions import basic_statistics, median_table, sort_labels
from htc.utils.JSONSchemaMeta import JSONSchemaMeta
from htc.utils.LabelMapping import LabelMapping
from htc.utils.Task import Task


def compress_html(file: Union[Path, None], fig_or_html: Union[go.Figure, str]) -> Union[str, None]:
    """
    Compress the Plotly figure as self-extracting HTML file. Like fig.write_html(), the result is also an HTML file which can be opened in every browser. But the file uses compression internally (gzip compression embedded as bas64) so the file size can be dramatically reduced if the original file size is large (e.g. when saving images).

    Note: If the compression resulted in a larger file size than the original file (e.g. when the figure is small), then the original HTML file is going to be used (i.e. without compression) so that this function never produces files which are larger than necessary.

    This is especially useful for saving overlay images (e.g. segmentation mask):
    >>> from htc.models.image.DatasetImage import DatasetImage
    >>> path = DataPath.from_image_name("P044#2020_02_01_09_51_15")
    >>> sample = DatasetImage([path], train=False, config=Config({"input/no_features": True}))[0]
    >>> fig = create_segmentation_overlay(sample['labels'].numpy(), path)
    >>> html = fig.to_html(include_plotlyjs="cdn", div_id='segmentation')

    Original file size in MiB:
    >>> len(html) / 2**20  # doctest: +ELLIPSIS
    5.10...
    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile() as tmpfile:
    ...    tmpfile = Path(tmpfile.name)
    ...    compress_html(tmpfile, html)
    ...    compressed_size = tmpfile.stat().st_size

    Compressed file size in MiB:
    >>> compressed_size / 2**20  # doctest: +ELLIPSIS
    0.63...

    Compression ratio:
    >>> compressed_size / len(html)  # doctest: +ELLIPSIS
    0.12...

    Args:
        file: Path to the output file. If none, the resulting HTML string will be returned.
        fig_or_html: Either a Plotly figure object or an HTML string.
    """

    # The following code is based on https://github.com/six-two/self-unzip.html/blob/main/python/self_unzip_html/__init__.py
    def compress(content: str) -> str:
        # mtime for reproducible results
        file_bytes = gzip.compress(content.encode(), mtime=0)
        file_bytes = base64.a85encode(file_bytes, adobe=True)
        file_bytes = file_bytes.replace(b'"', b"v").replace(b"\\", b"w")
        return file_bytes.decode()

    if isinstance(fig_or_html, go.Figure):
        assert file is not None, "You must provide a file if not providing the html"
        html_content = fig_or_html.to_html(include_plotlyjs="cdn", div_id=file.stem)
    else:
        html_content = fig_or_html
    compressed_data = compress(html_content)

    # From here: https://github.com/six-two/self-unzip.html/blob/main/python/self_unzip_html/template.html
    template = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Self Extracting Page</title>
</head>
<body>
    <h1>Unpacking...</h1>
    <p>Depending on the file size of the HTML document, unpacking may take a few seconds. If you can read this for a longer period of time, please make sure that you do not have any plugins which may block this page from running the extraction code (e.g. NoScript).</p>

    <script>
    // Contains minified code from:
    // - https://github.com/101arrowz/fflate, MIT License, Copyright (c) 2020 Arjun Barrett
    // - https://github.com/nE0sIghT/ascii85.js, MIT License, Copyright (C) 2018  Yuri Konotopov (Юрий Конотопов) <ykonotopov@gnome.org>
    (function(){function S(a,c=4){let b=new Uint8Array(c);for(let e=0;e<c;e++)b[e]=a>>T[e]&255;return b}function U(a){function c(){let d=S(g,p-1);for(let h=0;h<d.length;h++)b.push(d[h]);g=p=0}let b=[];var e=!1;let g=0,p=0,q=a.startsWith("<~")&&2<a.length?2:0;do if(0!==a.charAt(q).trim().length){var m=a.charCodeAt(q);switch(m){case 122:0!=p&&console.error("Unexpected 'z' character at position "+q);for(m=0;4>m;m++)b.push(0);break;case 126:e="";for(m=q+1;m<a.length&&0==e.trim().length;)e=a.charAt(m++);">"!=
    e&&console.error("Broken EOD at position "+m);p&&(g+=L[p-1],c());e=!0;break;default:(33>m||117<m)&&console.error("Unexpected character with code "+m+" at position "+q),g+=(m-33)*L[p++],5<=p&&c()}}while(q++<a.length&&!e);return new Uint8Array(b)}var r=Uint8Array,C=Uint16Array,M=Uint32Array,N=new r([0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0,0]),O=new r([0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,0,0]),V=new r([16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,
    15]),k=function(a,c){for(var b=new C(31),e=0;31>e;++e)b[e]=c+=1<<a[e-1];a=new M(b[30]);for(e=1;30>e;++e)for(c=b[e];c<b[e+1];++c)a[c]=c-b[e]<<5|e;return[b,a]},l=k(N,2),P=l[0];l=l[1];P[28]=258;l[258]=28;var W=k(O,0)[0],F=new C(32768);for(k=0;32768>k;++k)l=(k&43690)>>>1|(k&21845)<<1,l=(l&52428)>>>2|(l&13107)<<2,l=(l&61680)>>>4|(l&3855)<<4,F[k]=((l&65280)>>>8|(l&255)<<8)>>>1;var D=function(a,c,b){for(var e=a.length,g=0,p=new C(c);g<e;++g)a[g]&&++p[a[g]-1];var q=new C(c);for(g=0;g<c;++g)q[g]=q[g-1]+p[g-
    1]<<1;if(b)for(b=new C(1<<c),p=15-c,g=0;g<e;++g){if(a[g]){var m=g<<4|a[g],d=c-a[g],h=q[a[g]-1]++<<d;for(d=h|(1<<d)-1;h<=d;++h)b[F[h]>>>p]=m}}else for(b=new C(e),g=0;g<e;++g)a[g]&&(b[g]=F[q[a[g]-1]++]>>>15-a[g]);return b};l=new r(288);for(k=0;144>k;++k)l[k]=8;for(k=144;256>k;++k)l[k]=9;for(k=256;280>k;++k)l[k]=7;for(k=280;288>k;++k)l[k]=8;var Q=new r(32);for(k=0;32>k;++k)Q[k]=5;var X=D(l,9,1),Y=D(Q,5,1),G=function(a){for(var c=a[0],b=1;b<a.length;++b)a[b]>c&&(c=a[b]);return c},v=function(a,c,b){var e=
    c/8|0;return(a[e]|a[e+1]<<8)>>(c&7)&b},H=function(a,c){var b=c/8|0;return(a[b]|a[b+1]<<8|a[b+2]<<16)>>(c&7)},Z=function(a,c,b){if(null==c||0>c)c=0;if(null==b||b>a.length)b=a.length;var e=new (2==a.BYTES_PER_ELEMENT?C:4==a.BYTES_PER_ELEMENT?M:r)(b-c);e.set(a.subarray(c,b));return e},aa=["unexpected EOF","invalid block type","invalid length/literal","invalid distance","stream finished","no stream handler",,"no callback","invalid UTF-8 data","extra field too long","date not in range 1980-2099","filename too long",
    "stream finishing","invalid zip data"],w=function(a,c,b){c=Error(c||aa[a]);c.code=a;Error.captureStackTrace&&Error.captureStackTrace(c,w);if(!b)throw c;return c},K=function(a,c,b){var e=a.length;if(!e||b&&b.f&&!b.l)return c||new r(0);var g=!c||b,p=!b||b.i;b||={};c||=new r(3*e);var q=function(E){var R=c.length;E>R&&(E=new r(Math.max(2*R,E)),E.set(c),c=E)},m=b.f||0,d=b.p||0,h=b.b||0,t=b.l,y=b.d,z=b.m,x=b.n,I=8*e;do{if(!t){m=v(a,d,1);var f=v(a,d+1,3);d+=3;if(f)if(1==f)t=X,y=Y,z=9,x=5;else if(2==f){z=
    v(a,d,31)+257;y=v(a,d+10,15)+4;t=z+v(a,d+5,31)+1;d+=14;x=new r(t);var n=new r(19);for(f=0;f<y;++f)n[V[f]]=v(a,d+3*f,7);d+=3*y;f=G(n);y=(1<<f)-1;var J=D(n,f,1);for(f=0;f<t;)if(n=J[v(a,d,y)],d+=n&15,u=n>>>4,16>u)x[f++]=u;else{var B=n=0;16==u?(B=3+v(a,d,3),d+=2,n=x[f-1]):17==u?(B=3+v(a,d,7),d+=3):18==u&&(B=11+v(a,d,127),d+=7);for(;B--;)x[f++]=n}t=x.subarray(0,z);f=x.subarray(z);z=G(t);x=G(f);t=D(t,z,1);y=D(f,x,1)}else w(1);else{var u=((d+7)/8|0)+4;d=a[u-4]|a[u-3]<<8;f=u+d;if(f>e){p&&w(0);break}g&&q(h+
    d);c.set(a.subarray(u,f),h);b.b=h+=d;b.p=d=8*f;b.f=m;continue}if(d>I){p&&w(0);break}}g&&q(h+131072);u=(1<<z)-1;J=(1<<x)-1;for(B=d;;B=d){n=t[H(a,d)&u];f=n>>>4;d+=n&15;if(d>I){p&&w(0);break}n||w(2);if(256>f)c[h++]=f;else if(256==f){B=d;t=null;break}else{n=f-254;if(264<f){f-=257;var A=N[f];n=v(a,d,(1<<A)-1)+P[f];d+=A}f=y[H(a,d)&J];A=f>>>4;f||w(3);d+=f&15;f=W[A];3<A&&(A=O[A],f+=H(a,d)&(1<<A)-1,d+=A);if(d>I){p&&w(0);break}g&&q(h+131072);for(n=h+n;h<n;h+=4)c[h]=c[h-f],c[h+1]=c[h+1-f],c[h+2]=c[h+2-f],c[h+
    3]=c[h+3-f];h=n}}b.l=t;b.p=B;b.b=h;b.f=m;t&&(m=1,b.m=z,b.d=y,b.n=x)}while(!m);return h==c.length?c:Z(c,0,h)};k=new r(0);l="undefined"!=typeof TextDecoder&&new TextDecoder;try{l.decode(k,{stream:!0})}catch(a){}const T=[24,16,8,0],L=[52200625,614125,7225,85,1];decompress=a=>{dec=U(a);a=dec;if(31==a[0]&&139==a[1]&&8==a[2]){var c=a.subarray;31==a[0]&&139==a[1]&&8==a[2]||w(6,"invalid gzip data");var b=a[3],e=10;b&4&&(e+=
    a[10]|(a[11]<<8)+2);for(var g=(b>>3&1)+(b>>4&1);0<g;g-=!a[e++]);c=c.call(a,e+(b&2),-8);b=a.length;a=K(c,new r((a[b-4]|a[b-3]<<8|a[b-2]<<16|a[b-1]<<24)>>>0))}else 8!=(a[0]&15)||7<a[0]>>4||(a[0]<<8|a[1])%31?a=K(a,void 0):((8!=(a[0]&15)||7<a[0]>>>4||(a[0]<<8|a[1])%31)&&w(6,"invalid zlib data"),a[1]&32&&w(6,"invalid zlib data: preset dictionaries not supported"),a=K(a.subarray(2,-4),void 0));dec=a;return dec}})();

    // My code (c) six-two, MIT License
    const c_data = "{{DATA}}";
    const og_data = decompress(c_data.replaceAll("v", '"').replaceAll("w", "\\"));
    {{CODE}}
    </script>
</body>
</html>
"""

    html_encoded = template.replace("{{DATA}}", compressed_data)
    JS_REPLACE = (
        "const og_text=new TextDecoder().decode(og_data);window.onload=()=>{let"
        ' n=document.open("text/html","replace");n.write(og_text);n.close()}'
    )
    html_encoded = html_encoded.replace("{{CODE}}", JS_REPLACE)

    html = (
        html_encoded
        if len(bytes(html_encoded, encoding="utf-8")) < len(bytes(html_content, encoding="utf-8"))
        else html_content
    )

    if file is None:
        return html
    else:
        with file.open("w", encoding="utf-8") as f:
            f.write(html)


def add_figcaption(fig_html: str, title: str, caption: str) -> str:
    css_figcaption = """
figure {
    /* Same font as plotly */
    font-family: "Open Sans", verdana, arial, sans-serif;
    width: min-content;
    margin-bottom: 25px;
}
figure > figcaption {
    margin-left: 15px;
    margin-right: 15px;
}
figcaption {
    text-align: center;
    margin-top: 10px;
}
"""

    return f"""
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>{title}</title>
        <meta charset="utf-8">
        <style>
        {css_figcaption}
        </style>
    </head>
    <body>
        <figure>
            {fig_html}
            <figcaption>{caption}</figcaption>
        </figure>
    </body>
</html>"""


def visualize_dict(data: Union[dict, list]) -> None:
    """
    Interactive visualization of a Python dictionary (to be used in a Jupyter notebook). The interactive visualization also works in exported HTML notebooks.

    Args:
        data: Python dict to visualize.
    """

    json_str = json.dumps(data, cls=AdvancedJSONEncoder)

    # Embed renderjson code (from here: https://github.com/caldwell/renderjson)
    with (Path(__file__).parent / "renderjson.js").open() as f:
        renderjson = f.read()

    # Embed everything into the HTML (this way it also works in the exported HTML notebook)
    div_id = uuid.uuid4()
    display(HTML(f"""
<div id="{div_id}" style="height: auto; width:100%;"></div>
<script>
    {renderjson}
    renderjson.set_show_to_level(1);
    document.getElementById('{div_id}').appendChild(renderjson({json_str}));
</script>"""))


def create_running_metric_plot(df: pd.DataFrame, metric_name: str = "dice_metric") -> go.Figure:
    fig = go.Figure()

    fold_values = []

    for f, fold_name in enumerate(sorted(df["fold_name"].unique())):
        df_fold = df.query(f'fold_name == "{fold_name}"')
        values = df_fold[metric_name].values
        epochs = df_fold["epoch_index"].values

        valid = ~np.isnan(values)
        values = np.maximum.accumulate(values[valid])
        epochs = epochs[valid]
        fold_values.append(values)

        line = {"color": plotly.colors.DEFAULT_PLOTLY_COLORS[f % len(plotly.colors.DEFAULT_PLOTLY_COLORS)], "width": 2}
        marker = {"size": 6}
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=values,
                mode="lines+markers",
                name=fold_name,
                legendgroup=fold_name,
                line=line,
                marker=marker,
                opacity=0.5,
            )
        )

    fold_values = np.stack(fold_values)
    fold_values = np.mean(fold_values, axis=0)

    line = {"color": "black", "width": 2}
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=fold_values,
            mode="lines+markers",
            name="mean",
            legendgroup="mean",
            line=line,
            marker=marker,
            opacity=0.5,
        )
    )

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text=metric_name)
    fig.update_layout(height=400, width=1000)
    fig.update_layout(title_x=0.5, title_text="Running maximum across folds")
    return fig


def show_loss_chart(df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> None:
    loss_names = [c for c in df_train.columns if c not in ["fold_name", "epoch_index", "step"] and "_step" not in c]
    n_metrics = len(loss_names)
    ece_name = []
    if df_val is not None and "ece" in df_val:
        assert (
            df_train["epoch_index"].max() == df_val["epoch_index"].max()
        ), "train and validation do not have the same epochs"
        n_metrics += 1
        ece_name = ["ece_error"]

    fig = make_subplots(rows=n_metrics, cols=1, subplot_titles=loss_names + ece_name)

    for l, loss_name in enumerate(loss_names):
        for f, fold_name in enumerate(sorted(df_train["fold_name"].unique())):
            df_fold = df_train.query(f'fold_name == "{fold_name}"')
            losses = df_fold[loss_name].values
            epochs = df_fold["epoch_index"].values

            valid = ~np.isnan(losses)
            losses = losses[valid]
            epochs = epochs[valid]

            line = {
                "color": plotly.colors.DEFAULT_PLOTLY_COLORS[f % len(plotly.colors.DEFAULT_PLOTLY_COLORS)],
                "width": 2,
            }
            marker = {"size": 6}
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=losses,
                    mode="lines+markers",
                    name=fold_name,
                    legendgroup=fold_name,
                    showlegend=l == 0,
                    line=line,
                    marker=marker,
                    opacity=0.5,
                ),
                row=l + 1,
                col=1,
            )
            fig.update_xaxes(title_text="Epoch", row=l + 1, col=1)

    if df_val is not None and "ece" in df_val:
        # Calculate the ece error
        for f, fold_name in enumerate(sorted(df_val["fold_name"].unique())):
            df_fold = df_val.query(f'fold_name == "{fold_name}"')
            epochs = df_fold["epoch_index"].unique()

            ece_error = []
            for epoch in epochs:
                df_epoch = df_fold.query(f"epoch_index == {epoch}")

                # Collect all ece values from all images and calculate the ece error based on these values
                # This is more correct than averaging over the ece errors from all images
                acc_mat = np.stack([v["accuracies"] for v in df_epoch["ece"]])
                conf_mat = np.stack([v["confidences"] for v in df_epoch["ece"]])
                prob_mat = np.stack([v["probabilities"] for v in df_epoch["ece"]])
                ece = ECE.aggregate_vectors(acc_mat, conf_mat, prob_mat)
                ece_error.append(ece["error"])

            line = {
                "color": plotly.colors.DEFAULT_PLOTLY_COLORS[f % len(plotly.colors.DEFAULT_PLOTLY_COLORS)],
                "width": 2,
            }
            marker = {"size": 6}
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=ece_error,
                    mode="lines+markers",
                    name=fold_name,
                    legendgroup=fold_name,
                    showlegend=l == 0,
                    line=line,
                    marker=marker,
                    opacity=0.5,
                ),
                row=l + 2,
                col=1,
            )
            fig.update_xaxes(title_text="Epoch", row=l + 2, col=1)

    fig.layout.height = 200 * n_metrics
    fig.layout.width = 1000
    fig.update_layout(title_x=0.5, hovermode="x")
    fig.show()


def create_class_scores_figure(agg: MetricAggregation) -> None:
    df = agg.df
    mapping = LabelMapping.from_config(agg.config)
    class_dice = [[] for l in range(len(mapping))]
    pixel_counts = np.zeros(len(mapping))
    image_counts = np.zeros(len(mapping))

    # Collect all dice metric values from all images in all folds and separate the metric by classes
    n_images = 0

    if "fold_name" in df:
        df = df.query("epoch_index == best_epoch_index")

    for _, row in df.iterrows():  # Images
        for label_index, dice in zip(row["used_labels"], row["dice_metric"]):
            class_dice[label_index].append(dice)
            image_counts[label_index] += 1

        pixel_counts += np.sum(row["confusion_matrix"], axis=1)
        n_images += 1

    # Same label ordering for bar charts and box plot
    df_counts = pd.DataFrame({
        "label_name": mapping.label_names(),
        "n_pixels": pixel_counts,
        "n_images": image_counts,
    })
    df_counts = sort_labels(df_counts)
    colors = [settings.label_colors[name] for name in df_counts["label_name"]]

    fig = make_subplots(
        rows=3, cols=1, subplot_titles=["Total image counts", "Total pixel counts ", "Class dice across subjects"]
    )
    fig.add_trace(
        go.Bar(
            x=df_counts["label_name"],
            y=df_counts["n_images"],
            marker=dict(color=colors),
            showlegend=False,
            name="counts",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df_counts["label_name"],
            y=df_counts["n_pixels"],
            marker=dict(color=colors),
            showlegend=False,
            name="counts",
        ),
        row=2,
        col=1,
    )

    df_label = agg.grouped_metrics(keep_subjects=True)
    fig_box = px.box(
        df_label,
        x="label_name",
        y="dice_metric",
        color="label_name",
        points="all",
        hover_data=["subject_name"],
        color_discrete_map=settings.label_colors,
    )

    for trace in range(len(fig_box["data"])):
        fig.append_trace(fig_box["data"][trace], row=3, col=1)

    fig.update_yaxes(title_text="Counts", row=1, col=1)
    fig.update_yaxes(title_text="Counts", row=2, col=1)
    fig.update_yaxes(title_text="Dice metric", range=[-0.05, 1.05], row=3, col=1)
    fig.update_layout(height=700, width=1000)
    fig.show()


def show_class_scores_epoch(df: pd.DataFrame, mapping: LabelMapping) -> None:
    folds = sorted(df["fold_name"].unique())

    fig = make_subplots(rows=3, cols=1, subplot_titles=["Image counts", "Pixel counts", "Class dice"])
    button_states = (
        []
    )  # The buttons work by first drawing all plots and then switching of their visibility with the buttons
    line_ids = []
    colors = [settings.label_colors[name] for name in mapping.label_names()]

    for f, fold_name in enumerate(folds):
        df_fold = df.query(f'fold_name == "{fold_name}"')

        pixel_counts = np.zeros(len(mapping))
        image_counts = np.zeros(len(mapping))

        # Create a new table with the loss per class (over the epochs)
        rows = []
        epoch_indexs = df_fold["epoch_index"].unique()
        for epoch in epoch_indexs:
            df_epoch = df_fold.query(f"epoch_index == {epoch}")
            for i, row in df_epoch.iterrows():  # Images
                for label_index, dice in zip(row["used_labels"], row["dice_metric"]):
                    rows.append([epoch, mapping.index_to_name(label_index), dice])
                    image_counts[label_index] += 1

                pixel_counts += np.sum(row["confusion_matrix"], axis=1)

        df_epochs = pd.DataFrame(rows, columns=["epoch_index", "label", "dice"])

        pixel_counts /= len(epoch_indexs)
        image_counts /= len(epoch_indexs)

        # Count statistics per fold
        fig.add_trace(
            go.Bar(
                x=mapping.label_names(),
                y=image_counts,
                marker=dict(color=colors),
                showlegend=False,
                name="counts",
                visible=f == 0,
            ),
            row=1,
            col=1,
        )
        line_ids.append(f)
        fig.add_trace(
            go.Bar(
                x=mapping.label_names(),
                y=pixel_counts,
                marker=dict(color=colors),
                showlegend=False,
                name="counts",
                visible=f == 0,
            ),
            row=2,
            col=1,
        )
        line_ids.append(f)

        # Plot a line for each class label
        for label in df_epochs["label"].unique():
            df_label = df_epochs.query(f'label == "{label}"').sort_values(by=["epoch_index"])
            df_dice = df_label.groupby("epoch_index")["dice"]

            mean_dice = df_dice.mean().values
            min_dice = mean_dice - df_dice.min().values
            max_dice = df_dice.max().values - mean_dice

            fig.add_trace(
                go.Scatter(
                    x=df_label["epoch_index"].unique(),
                    y=mean_dice,
                    error_y={"type": "data", "symmetric": False, "array": max_dice, "arrayminus": min_dice},
                    mode="lines+markers",
                    name=label,
                    marker_color=settings.label_colors[label],
                    visible=f == 0,
                ),
                row=3,
                col=1,
            )

            line_ids.append(f)

        button_states.append({
            "label": fold_name,
            "method": "update",
            "args": [{"title": f"Statistics for the validation set ({fold_name})"}],
        })

    # Calculate the visible states (find out which lines have to be activated for which fold)
    line_ids = np.array(line_ids)
    for b, button_state in enumerate(button_states):
        visible_state = line_ids == b
        button_state["args"].insert(0, {"visible": visible_state.tolist()})

    fig.layout.update(updatemenus=[go.layout.Updatemenu(type="dropdown", active=0, buttons=button_states)])

    fig.update_yaxes(title_text="Counts", row=1, col=1)
    fig.update_yaxes(title_text="Counts", row=2, col=1)
    fig.update_yaxes(title_text="Dice metric", row=3, col=1)
    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    fig.layout.title = button_states[0]["args"][1]["title"]
    fig.update_layout(title_x=0.5)
    fig.layout.height = 900
    fig.layout.width = 1000
    fig.show()


def create_confusion_figure(confusion_matrix: np.ndarray, labels: list[str] = None) -> go.Figure:
    """Confusion matrix gives an impression of which classes are misclassified by which other classes."""
    if labels is None:
        labels = settings_seg.labels

    with np.errstate(invalid="ignore"):
        confusion_matrix_normalized = (confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True)) * 100

    hover_text = np.vectorize(lambda x: str(x))(confusion_matrix)

    data = go.Heatmap(
        z=confusion_matrix_normalized,
        y=labels,
        x=labels,
        text=hover_text,
        colorscale="Teal",
        colorbar={"title": "%", "thickness": 10},
        hovertemplate="true: %{y}<br>predicted: %{x}<br>row-ratio: %{z:.3f} %<br>pixels: %{text}",
    )
    annotations = []
    for i, row in enumerate(confusion_matrix):
        for j, value in enumerate(row):
            annotations.append({
                "x": labels[j],
                "y": labels[i],
                "font": {"color": "black" if value < 0.5 else "white"},
                "text": f"{value*100:.1f}",
                "xref": "x1",
                "yref": "y1",
                "showarrow": False,
            })

    layout = {"xaxis": {"title": "Predicted value"}, "yaxis": {"title": "Real value"}, "annotations": annotations}
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(len(confusion_matrix) * 50, 300), width=max(len(confusion_matrix) * 50, 300) + 100)
    fig.update_layout(title_x=0.5, title_text="Confusion matrix<br>(row-wise normalized)")

    return fig


def create_confusion_figure_comparison(
    cm_1: np.ndarray, cm_2: np.ndarray, run_dir_1: str, run_dir_2: str, labels: list[str] = None
) -> go.Figure:
    """Confusion matrix gives an impression of which classes are misclassified by which other classes."""
    if labels is None:
        labels = settings_seg.labels

    with np.errstate(invalid="ignore"):
        cm_1_norm = (cm_1 / np.sum(cm_1, axis=1, keepdims=True)) * 100
        cm_2_norm = (cm_2 / np.sum(cm_2, axis=1, keepdims=True)) * 100
        cm_normalized = cm_1_norm - cm_2_norm

    cm = cm_1 - cm_2

    hover_text = np.vectorize(lambda x: str(x))(cm_normalized)

    data = go.Heatmap(
        z=cm_normalized,
        y=labels,
        x=labels,
        text=hover_text,
        hovertemplate="true: %{y}<br>predicted: %{x}<br>row-ratio: %{z:.3f} %<br>pixels: %{text}",
    )
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append({
                "x": labels[j],
                "y": labels[i],
                "font": {"color": "white"},
                "text": f"{value:.4f}",
                "xref": "x1",
                "yref": "y1",
                "showarrow": False,
            })
    layout = {
        "xaxis": {"title": f"{run_dir_1} (rundir1)"},
        "yaxis": {"title": f"{run_dir_2} (rundir2)"},
        "annotations": annotations,
    }
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(title_x=0.5, title_text="rundir1 - rundir2")
    fig.layout.height = 800

    return fig


def discrete_colorbar(colors: list[tuple]) -> list[list[int, str]]:
    """
    Create a colormapping based on the list of discrete colors for Plotly.

    Args:
        colors: List of color tuples in the rgb format, e.g. (0.1, 0.5, 1.0) or rgba, e.g. (0.1, 0.5, 1.0, 0.8).

    Returns: Colormapping which can be passed to Plotly's colorscale argument.
    """
    assert len(colors) >= 1, "At least one color required"
    assert len(colors[0]) == 3 or len(colors[0]) == 4, "Exactly 3 (rgb) or 4 (rgba) color values required"
    assert all(all(0 <= v <= 1 for v in c) for c in colors), "All color values must be in the range [0;1]"

    # Create a color mapping according to plotlys documentation (list of normalized values and corresponding colors)
    color_mapping = []
    max_label = len(colors) - 1
    default_opacity = 1

    if max_label == 0:
        color = [int(c * 255) for c in colors[0]]
        opacity = default_opacity if len(color) == 3 else color[3]

        # Plotly requires at least a value for 0 and 1 (in this case they are the same since only one label exists in the image)
        color_mapping.append([0, f"rgba({color[0]},{color[1]},{color[2]}, {opacity})"])
        color_mapping.append([1, f"rgba({color[0]},{color[1]},{color[2]}, {opacity})"])
    else:
        # Make the colorbar discrete according to Plotly's specification: https://plotly.com/python/colorscales/#constructing-a-discrete-or-discontinuous-color-scale
        n_colors = max_label + 1
        mapping_values = np.linspace(0, 1, n_colors + 1)[1:-1]  # Values in-between [0.3, 0.6]
        mapping_values = np.sort(
            np.concatenate([mapping_values, mapping_values])
        )  # Duplicate values in-between [0.3, 0.3, 0.6, 0.6]
        mapping_values = np.append(
            np.insert(mapping_values, 0, 0), 1
        )  # Prepend with 0, append with 1 [0, 0.3, 0.3, 0.6, 0.6, 1]

        for i in range(n_colors):
            color = [int(c * 255) for c in colors[i]]
            opacity = default_opacity if len(color) == 3 else color[3]

            value_first = mapping_values[i * 2]
            color_mapping.append([value_first, f"rgba({color[0]},{color[1]},{color[2]},{opacity})"])

            value_second = mapping_values[i * 2 + 1]
            color_mapping.append([value_second, f"rgba({color[0]},{color[1]},{color[2]},{opacity})"])

    return color_mapping


def consecutive_segmentation(segmentation: np.ndarray, mapping: dict[int, int]):
    """
    Remap the label indices in segmentation to consecutive values as defined in the mapping. Every other value will be set to None. Required for heatmaps in Plotly with discrete colorbars.

    Args:
        segmentation: Image with label indices.
        mapping: Original label index to consecutive label index mapping.

    Returns: Segmentation image with consecutive label indices. dtype will be float to support None values.
    """
    assert list(mapping.values()) == list(range(len(mapping))), "The mapping must map to consecutive values"
    segmentation = tensor_mapping(segmentation.astype(np.int64), mapping)
    segmentation = segmentation.astype(np.float32)

    # Every value which is not part of the mapping denotes an invalid value
    # Invalid values break the color mapping and must be set explicitly to None for Plotly
    segmentation[segmentation >= len(mapping)] = None

    return segmentation


def dict_to_html_list(data: dict, schema: JSONSchemaMeta = None) -> str:
    """
    Simple function to display dict data as a nested HTML list.

    Args:
        data: The data to display.
        schema: Optional json schema reference with meta information about the data (currently, the keys `unit`, `level_of_measurement` and `description` are supported). This information is usually stored in the `meta.schema` file of the data folder.

    Returns: The HTML string with the nested list.
    """
    html = "<ul>"
    for k, v in data.items():
        schema_k = schema[k] if schema is not None else None

        if type(v) == list:
            v_html = ", ".join([str(e) for e in v])
        elif type(v) == dict:
            v_html = dict_to_html_list(v, schema_k)
        else:
            v_html = str(v)

            # Add infos based on the associated schema
            if schema_k is not None:
                if (unit := schema_k.meta("unit")) is not None:
                    v_html += f" {unit}"
                if (level := schema_k.meta("level_of_measurement")) is not None:
                    v_html += (
                        ' [<a href="https://en.wikipedia.org/wiki/Level_of_measurement" target="_blank" title="Level'
                        f' of measurement (scale of measure)">{level}-scaled</a>]'
                    )

        if schema_k is not None and schema_k.has_meta("description"):
            k_text = f'<abbr title="{schema_k.meta("description")}">{k}</abbr>'
        else:
            k_text = k

        html += f"<li>{k_text}: {v_html}</li>"
    html += "</ul>"

    return html


def create_median_spectra_figure(path: DataPath) -> go.Figure:
    """
    Create a median spectra plot with all available labels and from all annotators.

    Args:
        path: Data path to the image.

    Returns: Plotly figure.
    """
    label_mapping = LabelMapping.from_path(path)

    df = median_table(image_names=[path.image_name()], annotation_name="all")
    df = sort_labels(df)
    df = df.query("label_name in @path.annotated_labels('all')")
    line_options = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot", "5, 10, 5", "2, 10, 2", "5, 2, 5"]
    annotator_mapping = {a: line_options[i] for i, a in enumerate(df["annotation_name"].unique())}

    fig = go.Figure()
    for i, row in df.iterrows():
        add_std_fill(
            fig,
            row["median_normalized_spectrum"],
            row["std_normalized_spectrum"],
            x=tivita_wavelengths(),
            linecolor=label_mapping.name_to_color(row["label_name"]),
            label=row["label_name"] + ": " + row["annotation_name"],
            line_dash=annotator_mapping[row["annotation_name"]],
        )

    fig.update_layout(width=1000, height=500, template="plotly_white")
    fig.update_layout(xaxis_title="wavelength [nm]", yaxis_title="normalized reflectance [a.u.]")
    fig.update_layout(
        title_x=0.5,
        title_text="Median spectra per label (shaded area represents one standard deviation around the median)",
    )

    return fig


def create_median_spectra_comparison_figure(
    df: pd.DataFrame,
    group_column: str,
    color_mapping: dict[str, str],
    line_dash_mapping: dict[str, str] = None,
) -> go.Figure:
    """
    Compare median spectra for different groups (e.g. cameras) stratified by label.

    Args:
        df: Dataframe with the median spectra for all groups and labels.
        group_column: Column name which contains the group names.
        color_mapping: Mapping from group names to colors.
        line_dash_mapping: Mapping from group names to line dash styles. If None, all lines will be solid.

    Returns: Plotly figure.
    """
    n_cols = 4
    n_rows = math.ceil(df["label_name"].nunique() / n_cols)
    n_missing = n_cols * n_rows - df["label_name"].nunique()
    labels = df["label_name"].unique()
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes="all" if n_missing == 0 else False,
        shared_yaxes="all",
        subplot_titles=labels,
        vertical_spacing=0.05,
        horizontal_spacing=0.03,
    )

    for group_name in df[group_column].unique():
        for i, label in enumerate(labels):
            df_cam = df[(df["label_name"] == label) & (df[group_column] == group_name)]
            if len(df_cam) > 0:
                row = i // n_cols
                col = i % n_cols

                spec = np.stack(df_cam["median_normalized_spectrum"])  # Spectra from all subjects
                spec_mean = np.mean(spec, axis=0)
                spec_std = np.std(spec, axis=0)

                if line_dash_mapping is None:
                    line_dash = "solid"
                else:
                    line_dash = line_dash_mapping[group_name]

                fig = add_std_fill(
                    fig,
                    spec_mean,
                    spec_std,
                    x=tivita_wavelengths(),
                    linecolor=color_mapping[group_name],
                    line_dash=line_dash,
                    label=group_name,
                    showlegend=row == 1 and col == 1,
                    row=row + 1,
                    col=col + 1,
                )

                if col == 0:
                    fig.update_yaxes(title="<b>L1 normalized<br>reflectance [a.u.]</b>", row=row + 1, col=col + 1)
                if row == n_rows - 1:
                    fig.update_xaxes(title="wavelength [nm]", row=row + 1, col=col + 1)

    if n_missing != 0:
        # Manually add the x-axis ticks to the plots in the last rows
        fig.update_xaxes(showticklabels=False)
        ticks = [600, 700, 800, 900]
        for i in range(n_missing):
            fig.update_xaxes(
                tickmode="array",
                tickvals=ticks,
                ticktext=ticks,
                showticklabels=True,
                title="<b>wavelength [nm]</b>",
                row=n_rows - 1,
                col=n_cols - i,
            )
        for i in range(n_cols - n_missing):
            fig.update_xaxes(
                tickmode="array",
                tickvals=ticks,
                ticktext=ticks,
                showticklabels=True,
                title="<b>wavelength [nm]</b>",
                row=n_rows,
                col=i + 1,
            )

    if n_missing != 0:
        # Manually add the x-axis ticks to the plots in the last rows
        fig.update_xaxes(showticklabels=False)
        ticks = [600, 700, 800, 900]
        for i in range(n_missing):
            fig.update_xaxes(
                tickmode="array",
                tickvals=ticks,
                ticktext=ticks,
                showticklabels=True,
                title="<b>wavelength [nm]</b>",
                row=n_rows - 1,
                col=n_cols - i,
            )
        for i in range(n_cols - n_missing):
            fig.update_xaxes(
                tickmode="array",
                tickvals=ticks,
                ticktext=ticks,
                showticklabels=True,
                title="<b>wavelength [nm]</b>",
                row=n_rows,
                col=i + 1,
            )

    fig.update_layout(
        title="Spectra for organs and cameras (with inter-pig deviation)",
        title_x=0.5,
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=70))
    fig.update_layout(height=200 * n_rows, width=1400)
    fig.update_layout(template="plotly_white")

    return fig


def create_overview_document(
    path: DataPath,
    include_tpi: bool = False,
    navigation_paths: list[DataPath] = None,
    navigation_link_callback: Callable[[str, str, DataPath], str] = None,
    nav_width: str = "23em",
    searchable_meta_attributes: list[str] = None,
) -> str:
    """
    Create an overview figure for the given image. It will show the RGB image with all the available annotations plus the tissue parameter images.

    Args:
        path: Data path to the image.
        include_tpi: If True, TPI images are included in the overview document (adds around 10 MiB to the output).
        navigation_paths: If not None, will add a navigation bar with all links sorted by organ. The user can use this navigation bar to easily switch between images.
        navigation_link_callback: Callback which receives the label name, number and data path of the target image and should create a relative link where the corresponding local html file for the target image can be found. If parts of the link contain invalid URL characters (e.g. # in image name), then please wrap it in quote_plus before (e.g. quote_plus(p.image_name())). For example, ('spleen', '08', DataPath) --> '../08_spleen/P086%232021_04_15_09_22_20.html'.
        nav_width: Width of the navigation bar (in CSS units).
        searchable_meta_attributes: List of meta attributes which should be searchable. If None, the annotation_name will be searchable per default. You need to include the annotation_name yourself if you change this parameter.

    Returns: HTML string which is best saved with the `compress_html()` function.
    """
    if searchable_meta_attributes is None:
        searchable_meta_attributes = ["annotation_name"]

    seg = path.read_segmentation(annotation_name="all")
    if seg is None or len(path.annotated_labels(annotation_name="all")) == 0:
        # No annotations available, only show the RGB image
        rgb_image = path.read_rgb_reconstructed()
        fig_seg = px.imshow(rgb_image)

        # Similar size as create_segmentation_overlay()
        img_height, img_width = rgb_image.shape[:2]
        fig_seg.update_layout(height=img_height * 1.5, width=img_width * 1.53, template="plotly_white")
        fig_seg.update_layout(title_x=0.5, title_text=path.image_name())
        fig_median = None
    else:
        fig_seg = create_segmentation_overlay(seg, path)
        fig_median = create_median_spectra_figure(path)

    # Remove the Plotly title because it cannot be selected
    # We'll add the title manually via HTML below
    fig_seg.update_layout(margin=dict(t=0), title=None)

    if include_tpi:
        images = [path.compute_sto2().data, path.compute_nir().data, path.compute_ohi().data, path.compute_twi().data]
        names = [
            "Tissue oxygen saturation (StO2)",
            "NIR Perfusion Index",
            "Organ Hemoglobin Index (OHI)",
            "Tissue Water Index (TWI)",
        ]
        colorscale = tivita_colorscale("plotly")
        fig_tpi = make_subplots(
            2,
            2,
            shared_xaxes="all",
            shared_yaxes="all",
            subplot_titles=names,
            horizontal_spacing=0.02,
            vertical_spacing=0.05,
        )

        fig_tpi.add_trace(go.Heatmap(z=images[0], coloraxis="coloraxis", name="Sto2"), row=1, col=1)
        fig_tpi.add_trace(go.Heatmap(z=images[1], coloraxis="coloraxis", name="NIR"), row=1, col=2)
        fig_tpi.add_trace(go.Heatmap(z=images[2], coloraxis="coloraxis", name="OHI"), row=2, col=1)
        fig_tpi.add_trace(go.Heatmap(z=images[3], coloraxis="coloraxis", name="TWI"), row=2, col=2)

        fig_tpi.update_layout(coloraxis=dict(colorscale=colorscale), showlegend=False)
        fig_tpi.update_layout(yaxis_autorange="reversed")
        fig_tpi.update_layout(width=1000, height=800)

    skip_keys = {"dsettings", "dataset_env_name", "data_dir", "intermediates_dir"}
    image_meta = {k: v for k, v in path.meta().items() if k not in skip_keys}
    if len(image_meta) > 0:
        meta_html = f"<h3>Meta annotation for the image {path.image_name()}:</h3>\n"
        meta_html += dict_to_html_list(image_meta, schema=path.annotation_meta_schema())
    else:
        meta_html = ""

    if navigation_paths is not None:
        assert (
            navigation_link_callback is not None
        ), "navigation_link_callback must be provided if a navigation pane should be created"

        # Use the label ordering if available
        masks_settings = DatasetSettings(settings.data_dirs.masks)
        if "label_ordering" in path.dataset_settings:
            label_ordering = path.dataset_settings.get("label_ordering", {})
        else:
            label_ordering = masks_settings.get("label_ordering", {})

        # All paths per organ (there may be more than one organ per image)
        image_labels = path.annotated_labels(annotation_name="all")
        label_paths = {}
        used_paths = []
        for p in navigation_paths:
            labels = p.annotated_labels(annotation_name="all")

            # We only create overview files for images which have at least one label
            if len(labels) > 0:
                used_paths.append(p)

            for l in labels:
                if l not in label_paths:
                    label_paths[l] = []
                label_paths[l].append(p)

        label_paths = sort_labels(label_paths)

        # Previous/Next image according to the sorting of the navigation paths
        prev_image = None
        next_image = None
        for i, p in enumerate(used_paths):
            if p == path:
                if i > 0:
                    prev_path = used_paths[i - 1]
                    label_name = prev_path.annotated_labels(annotation_name="all")[0]
                    prev_image = {
                        "path": prev_path,
                        "label_name": label_name,
                        "label_number": label_ordering.get(label_name, None),
                    }
                if i < len(used_paths) - 1:
                    next_path = used_paths[i + 1]
                    label_name = next_path.annotated_labels(annotation_name="all")[0]
                    next_image = {
                        "path": next_path,
                        "label_name": label_name,
                        "label_number": label_ordering.get(label_name, None),
                    }

        details_html = ""
        link_index = 0  # Global index for each link in the list (for scrolling)
        for l in label_paths.keys():
            label_paths[l] = sorted(label_paths[l])

            # Use the label ordering if available
            label_number = label_ordering.get(l, None)

            # List with links to all paths of the current label
            paths_html = ""
            for p in label_paths[l]:
                selected = 'class="selected" ' if p == path else ""

                label_meta = p.meta(f"label_meta/{l}")
                if label_meta is not None and "situs" in label_meta:
                    meta = f"<br>situs={label_meta['situs']}"
                    if "angle" in label_meta:
                        meta += f", angle={label_meta['angle']}°"
                    if "repetition" in label_meta:
                        meta += f", repetition={label_meta['repetition']}"
                else:
                    meta = ""

                invisible_meta = ""
                for attribute in searchable_meta_attributes:
                    attribute_meta = p.meta(attribute)
                    if attribute_meta is not None:
                        if type(attribute_meta) == list:
                            attribute_meta = " ".join(attribute_meta)
                        invisible_meta += f'<span class="invisible">{attribute}={attribute_meta} </span>'

                link = navigation_link_callback(l, label_number, p) + f"?nav=show&link_index={link_index}"
                paths_html += (
                    f'<li><a id="link_{link_index}"'
                    f' {selected}href="{link}">{p.image_name()}{meta}{invisible_meta}</a></li>'
                )
                link_index += 1

            # Add an image for the current label if available
            svg_path = path.data_dir / "extra_label_symbols" / f"Cat_{label_number}_{l}.svg"
            if not svg_path.exists():
                # Try to find the label symbol in the masks dataset
                svg_path = settings.data_dirs.masks / "extra_label_symbols" / f"Cat_{label_ordering.get(l, '')}_{l}.svg"
            if not svg_path.exists():
                # In case the label ordering is different, try to find the symbol by name in the masks dataset
                svg_files = sorted((settings.data_dirs.masks / "extra_label_symbols").glob("*.svg"))
                for f in svg_files:
                    if re.search(r"Cat_\d+_" + l, f.stem) is not None:
                        svg_path = f
                        break

            if svg_path.exists():
                with svg_path.open("r") as f:
                    svg = f.read()
                    svg_base64 = base64.b64encode(svg.encode()).decode()
                label_image = f" <img class='label_image' src='data:image/svg+xml;base64,{svg_base64}' />"
            else:
                label_image = ""

            selected = ' class="selected"' if l in image_labels else ""
            details_open = " open" if l in image_labels else ""
            if label_number is not None:
                summary_name = f"{label_number}_{l}{label_image}"
            else:
                summary_name = f"{l}{label_image}"

            details_html += (
                f"<details id='{l}'"
                f" {details_open}><summary{selected}>{summary_name}</summary><ul>{paths_html}</ul></details>"
            )

        if prev_image is not None:
            prev_link = navigation_link_callback(
                prev_image["label_name"], prev_image["label_number"], prev_image["path"]
            )
            prev_link = (
                f'<a class="prev_next" href="{prev_link}" title="Previous image (according to the sorting of the'
                f' filesystem): {prev_image["path"].image_name()}">« previous</a>'
            )
        else:
            prev_link = (
                '<span class="prev_next no_link" title="No previous image available (first image of the dataset)">«'
                " previous</span>"
            )
        if next_image is not None:
            next_link = navigation_link_callback(
                next_image["label_name"], next_image["label_number"], next_image["path"]
            )
            next_link = (
                f'<a class="prev_next" href="{next_link}" title="Next image (according to the sorting of the'
                f' filesystem): {next_image["path"].image_name()}">next »</a>'
            )
        else:
            next_link = (
                '<span class="prev_next no_link" title="No next image available (last image of the dataset)">«'
                " next</span>"
            )

        search_function = """
function searchAll() {
    // We first search all details elements (labels) and then all list elements (images)
    const needle = document.getElementById("nav_search").value.toLowerCase();
    document.querySelectorAll("#image_navigation details").forEach(function (detailsElement) {
        if (detailsElement.textContent.toLowerCase().includes(needle)) {
            detailsElement.style.display = "";

            // Show only the images which match
            // If no image matches (e.g. search for organs), show all
            let imageList = detailsElement.getElementsByTagName("ul")[0];
            if (imageList.textContent.toLowerCase().includes(needle)) {
                for (let listElement of imageList.getElementsByTagName("li")) {
                    if (listElement.textContent.toLowerCase().includes(needle)) {
                        listElement.style.display = "";
                    } else {
                        listElement.style.display = "none";
                    }
                }
            }
        } else {
            detailsElement.style.display = "none";
        }
    });

    // Find out which elements of the list are visible
    let nVisible = 0;
    document.querySelectorAll("#image_navigation details").forEach(function (detailsElement) {
        if (detailsElement.style.display != "none") {
            for (let listElement of detailsElement.getElementsByTagName("li")) {
                if (listElement.style.display != "none") {
                    nVisible++;
                }
            }
        }
    });

    // If filtering is active, display the number of visible elements
    let searchCount = document.getElementById("search_count");
    const allListElements = document.querySelectorAll("#image_navigation li");
    searchCount.innerText = nVisible + " of " + allListElements.length + " elements shown";
    if (nVisible < allListElements.length) {
        searchCount.style.display = "";
    } else {
        searchCount.style.display = "none";
    }
}
"""
        nav_html = f"""
<span id="nav_link" onclick="openNav()">&#9776; Image selection</span>{prev_link}{next_link}

<nav id="image_navigation">
  <a href="javascript:void(0)" class="closebtn" onclick="closeNav()">&times;</a>
  <input type="text" id="nav_search" onkeyup="searchAll()" placeholder="Search all.." title="Search for everything which is visible in the navigation panel plus the following invisible attributes: {', '.join(searchable_meta_attributes)}">
  <p id="search_count"></p>
  {details_html}
</nav>
<script>
{search_function}
function openNav() {{
  document.getElementById("image_navigation").style.width = "{nav_width}";
}}

function closeNav() {{
  document.getElementById("image_navigation").style.width = "0px";
}}

document.addEventListener('DOMContentLoaded', function() {{
  // Set previous navigation state
  const urlParams = new URLSearchParams(location.search);
  if (urlParams.has("nav") && urlParams.get("nav") == "show") {{
    openNav();
  }}

  let isNavOpen = false;
  let initialScroll = false;  // Scroll to the selected link on page load
  document.getElementById("image_navigation").addEventListener('transitionend', function() {{
    isNavOpen = document.getElementById("image_navigation").style.width != "0px";

    if (!initialScroll && urlParams.has("link_index")) {{
      let current_link = document.getElementById("link_" + urlParams.get("link_index"));
      current_link.scrollIntoView({{behavior: "smooth", block: "center", inline: "nearest"}});
      initialScroll = true;
    }}
  }});

  window.addEventListener('click', function(e) {{
    if (isNavOpen && !document.getElementById('image_navigation').contains(e.target)) {{
      // If the user clicks outside the navigation bar, the navigation should close
      closeNav();
    }}
  }});

  if (urlParams.has("search_text")) {{
    // Apply an existing search text to the navigation bar
    document.getElementById("nav_search").value = urlParams.get("search_text");
    searchAll();
  }}

  // Pass the current search text to every link in the navigation bar
  const changeURL = function(event) {{
    // Stop the link from redirecting
    event.preventDefault();

    // Redirect instead with JavaScript
    const needle = document.getElementById("nav_search").value.toLowerCase();
    if (needle != "") {{
        let url = new URL(event.target.href);
        url.searchParams.set("search_text", needle);
        return url.href;
    }} else {{
        return event.target.href;
    }}
  }};

  // We need to manually handle click (and middle click) events for the links
  for (let link of document.querySelectorAll("#image_navigation a")) {{
    link.addEventListener('click', function(event) {{
        // Open link in the same window
        const newURL = changeURL(event);
        window.location.href = newURL;
    }}, false);
    link.addEventListener('auxclick', function(event) {{
        if (event.button == 1) {{
            // Middle click (open page in a new window)
            const newURL = changeURL(event);
            window.open(newURL, '_blank');
        }}
    }}, false);
  }}
}});
</script>
"""

        nav_css = """
<style>
nav {
  height: 100%;
  width: 0;
  position: fixed;
  z-index: 1;
  top: 0;
  right: 0;
  background-color: #f7f7f7;
  box-shadow: 2px 0px 5px #9b9b9b;
  transition: 0.5s;
  overflow-x: scroll;
  padding-top: 30px;
  padding-left: 5px;
  /* Move the scrollbar to the left side */
  direction: rtl;
}

/* The remaining elements should still be left-aligned */
nav * {
    direction: ltr;
}
#nav_search {
    display: block;
    margin-right: auto;
}
#nav_search, #search_count {
    margin-left: 8px;
    margin-bottom: 5px;
}

#prev_image {
  margin-left: 5px;
}

#next_image {
  position: absolute;
  right: 0;
  margin-right: 5px;
}

nav > details > summary {
  padding: 2px 8px 2px 8px;
  text-decoration: none;
  font-size: 1.3em;
  color: #818181;
  transition: 0.3s;
  cursor: pointer;
}

nav > details:last-child {
    /* Leave enough space at the bottom so that the user knows when the list ends */
    padding-bottom: 60px;
}

nav ul {
  margin-top: 0;
  margin-bottom: 0;
}

nav a {
  color: gray;
  text-decoration: none;
}

nav a:hover {
  text-decoration: underline;
}

nav details a {
  /* Leave enough space on the right so that it is still possible to select image names */
  margin-right: 2em;
}

nav .closebtn {
  position: absolute;
  top: 0;
  right: 15px;
  font-size: 36px;
  margin-left: 50px;
}

.selected {
  font-weight: bold;
}

.label_image {
  width: 1.5em;
  position: relative;
  top: 0.35em;
}

.invisible {
    display: none;
}

a {
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
}
#nav_link {
  font-size: 2em;
  cursor: pointer;
  padding-right: 10px;
}
#nav_link:hover {
    text-decoration: underline;
}
.prev_next {
    font-size: 1.5em;
    padding-right: 10px;
}
.no_link {
    color: #aaa;
}

@media screen and (max-height: 450px) {
  nav {padding-top: 15px;}
  nav a {font-size: 18px;}
}
</style>
"""
    else:
        nav_css = ""
        nav_html = ""

    # Same styling as Plotly
    css = """
<style>
body {
    font-family: "Open Sans", verdana, arial, sans-serif;
    font-size: 12px;
    fill: rgb(42, 63, 95);
    fill-opacity: 1;
}

#image_container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
#image_container > h2 {
    margin-bottom: 0;
    text-align: center;
}
</style>
    """

    html = f"""
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Overview for the image {path.image_name()}</title>
        {css}
        {nav_css}
    </head>
    <body>
        {nav_html}
        <div id="image_container">
            <h2>{path.image_name()}</h2>
            {fig_seg.to_html(full_html=False, include_plotlyjs='cdn', div_id='segmentation')}
        </div>
        {fig_median.to_html(full_html=False, include_plotlyjs='cdn', div_id='median_spectra') if fig_median is not None else ""}
        {meta_html}
        {fig_tpi.to_html(full_html=False, include_plotlyjs='cdn', div_id='tpi_images') if include_tpi else ""}
    </body>
</html>"""

    return html


def create_segmentation_overlay(
    segmentation: Union[np.ndarray, dict[str, np.ndarray]],
    path: DataPath = None,
    rgb_image: np.ndarray = None,
    label_mapping: LabelMapping = None,
) -> go.Figure:
    """
    Overlays a segmentation result over an RGB image with an opacity slider.

    Works only for the original mask files and not on predicted segmentations. `prediction_figure_html()` can be used in this case.

    Args:
        segmentation: Segmentation image with a label index for each image. Either of shape (H, W) or (2, H, W) in case of a multi-layered segmentation mask. If a dict with names as keys and segmentations as values, a button element will be added allowing to switch between the segmentations (useful to plot the annotations from different annotators).
        path: Path to the file image file. Used to get the name of the image, the label mapping of the dataset or the rgb image (if not provided explicitly).
        rgb_image: Image data which is shown in the background.
        label_mapping: The label mapping which defines how the values in the segmentation image should be interpreted. This is also used to get the colors and names for the labels.

    Returns: Plotly figure object (layout properties can still be adjusted).
    """
    if type(segmentation) == np.ndarray:
        segmentation = {"name": segmentation}

    # Load original image
    if rgb_image is None:
        assert path is not None, "A path is required if no rgb file is given"
        rgb_image = path.read_rgb_reconstructed()

    if label_mapping is None:
        assert path is not None, "A path is required if no label mapping is given"
        label_mapping = LabelMapping.from_path(path)

    # Construct segmentation layers
    layer1 = {}
    layer2 = {}
    for name, seg in segmentation.items():
        assert np.issubdtype(seg.dtype, np.integer), f"Segmentation must be integer type ({seg.dtype = }, {path = })"

        if not label_mapping.is_index_valid(seg).any():
            # If there are no valid labels in the segmentation mask, we cannot show it
            continue

        if seg.ndim == 3:
            assert seg.shape[0] == 2, f"Can only handle two-layer segmentations (not {seg.shape[0]})"
            layer1[name] = seg[0]
            layer2[name] = seg[1]
            spatial_shape = seg.shape[1:]
        else:
            layer1[name] = seg
            spatial_shape = seg.shape

        assert spatial_shape == rgb_image.shape[:2], "The segmentation image must have the same shape as the RGB image"

    assert len(layer1) > 0, f"No valid segmentation found for the image {path}"

    img_height, img_width = rgb_image.shape[:2]
    opacity = 0.5
    fig = go.Figure()

    def label_remapping(seg: np.ndarray) -> tuple:
        # Color mapping based on the valid labels which occur in the image
        available_label_indices = np.unique(seg)
        label_mapping_valid = {
            label_index: label_mapping.index_to_name(label_index)
            for label_index in label_mapping.label_indices()
            if label_index in available_label_indices
        }
        label_colors = [to_rgba(label_mapping.name_to_color(label)) for label in label_mapping_valid.values()]

        # Create a remapping based on the labels which occur in the image (we need consecutive indices for the colormapping)
        remapping_minimal = {label_index: i for i, label_index in enumerate(label_mapping_valid.keys())}

        return label_mapping_valid, label_colors, remapping_minimal

    def get_colorbar(label_mapping_valid: dict[int, str]) -> dict:
        if len(label_mapping_valid) == 1:
            tickvals = [0, 1]
        else:
            assert (
                len(label_mapping_valid) > 0
            ), f"No valid labels found for the image:\n{path = }\n{path.annotated_labels(annotation_name='all') = }"

            # For example, with three colors, Plotly uses 0, 1, 2 and the colors change at 2/3, 4/3
            # We want the ticks to be placed in the middle of the color rectangles 2/6=2/3*0.5, 6/6=2/6+2/3
            tickstep = (len(label_mapping_valid) - 1) / len(label_mapping_valid)
            tickvals = np.arange(tickstep / 2, len(label_mapping_valid), tickstep)

        return {
            "title": "class",
            "tickvals": tickvals,
            "ticktext": list(label_mapping_valid.values()),
            "ticks": "outside",
            "yanchor": "top",
            "y": 1.01,
        }

    default_annotation_name = path.dataset_settings.get("annotation_name_default", "default")
    default_index = (
        0 if default_annotation_name not in layer1.keys() else list(layer1.keys()).index(default_annotation_name)
    )

    # The buttons hide or unhide individual plots but the number of plots per button state (=annotation_name) could be different due to multi-layer segmentations
    # Plotly only sees a global list of all plots (heatmaps in this case) and we need to tell Plotly which plots should be hidden or unhidden for each button state
    plot_index = 0  # Global plot index
    name_plot_mapping = {name: [] for name in layer1.keys()}  # List of global plot indices for each annotation name
    for i, name in enumerate(layer1.keys()):
        visible = i == default_index

        # We need a colorbar per annotation since there is no gurantee that all annotators used the same labels (and having one colorbar with all labels is not possible in Plotly)
        label_mapping_valid, label_colors, remapping_minimal = label_remapping(layer1[name])
        colorbar = get_colorbar(label_mapping_valid)

        if name in layer2:
            label_mapping_valid2, label_colors2, remapping_minimal2 = label_remapping(layer2[name])
            colorbar2 = get_colorbar(label_mapping_valid2)
            colorbar2["x"] = 1.25

            def create_text(label: int, label2: int) -> str:
                text = f"layer1: {label_mapping.index_to_name(label)}<br>"
                text += f"layer2: {label_mapping.index_to_name(label2)}"
                return text

            hover_text_ref = np.vectorize(create_text)(layer1[name], layer2[name])

            # Second segmentation layer with its own colorbar
            fig.add_trace(
                go.Heatmap(
                    z=np.flipud(consecutive_segmentation(layer2[name], remapping_minimal2)),
                    opacity=opacity,
                    colorscale=discrete_colorbar(label_colors2),
                    colorbar=colorbar2,
                    name="Segmentation",
                    visible=visible,
                )
            )
            name_plot_mapping[name].append(plot_index)
            plot_index += 1
        else:
            hover_text_ref = np.vectorize(lambda label: label_mapping.index_to_name(label))(layer1[name])

        fig.add_trace(
            go.Heatmap(
                z=np.flipud(consecutive_segmentation(layer1[name], remapping_minimal)),
                opacity=opacity,
                colorscale=discrete_colorbar(label_colors),
                colorbar=colorbar,
                text=np.flipud(hover_text_ref),
                name="Segmentation",
                visible=visible,
            )
        )
        name_plot_mapping[name].append(plot_index)
        plot_index += 1

    # We only need buttons if there are multiple annotation names
    button_states = [] if len(name_plot_mapping) > 1 else None
    if button_states is not None:
        for name, indices in name_plot_mapping.items():
            visible_state = [False] * plot_index  # The global visible state for this annotation name
            for i in indices:
                visible_state[i] = True

            button_states.append({
                "label": name,
                "method": "update",
                "args": [{"visible": visible_state}],
            })

    fig.add_layout_image(
        source=Image.fromarray(rgb_image),
        xref="x",
        yref="y",
        x=-0.5,
        y=img_height - 0.5,
        sizex=img_width,
        sizey=img_height,
        sizing="stretch",
        layer="below",
    )

    # Create and add slider
    steps = []
    opacity_range = np.arange(0, 1 + 0.05, 0.05)
    for o in opacity_range:
        steps.append({"method": "restyle", "args": [{"opacity": o}], "label": f"{o:.02f}"})

    opacity_slider = {
        "active": np.where(opacity_range == opacity)[0].item(),
        "currentvalue": {"prefix": "Opacity: "},
        "steps": steps,
    }

    if button_states is not None:
        fig.layout.update(
            updatemenus=[go.layout.Updatemenu(type="buttons", active=default_index, buttons=button_states)]
        )

    # Having a nice scaling here is a bit tricky since it depends on the width of the whole image which in return depends on the elements on the left and right (e.g. buttons, colorbars)
    scale_annotator = 1 if len(segmentation) == 1 else 1.12
    fig.update_layout(
        height=img_height * 1.5, width=img_width * 1.53 * scale_annotator, template="plotly_white", margin=dict(t=40)
    )
    fig.update_layout(
        sliders=[opacity_slider], title_x=0.5, title_y=0.98, title_text=path.image_name() if path is not None else ""
    )
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))  # Keep the aspect ratio while zooming
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    return fig


def create_overlay(overlay: np.ndarray, path: DataPath) -> go.Figure:
    """
    General function to overlay an image with a heatmap.

    Args:
        overlay: Array with the same shape as the image. The values are used to color the image.
        path: Path to the image.

    Returns: Plotly figure showing the overlay.
    """
    # Load original image
    rgb_image = path.read_rgb_reconstructed()
    assert overlay.shape == rgb_image.shape[:2], "The overlay image must have the same shape as the RGB image"
    rgb_image = Image.fromarray(rgb_image)

    img_height, img_width = path.dataset_settings["shape"][:2]
    opacity = 0.5
    fig = go.Figure()

    valid_values = np.unique(overlay[~np.isnan(overlay)])
    colorbar = {
        "title": "value",
        "ticks": "outside",
        "yanchor": "top",
        "y": 1.01,
        "tickvals": valid_values.tolist(),
        "ticktext": valid_values.tolist(),
    }

    colors = generate_distinct_colors(len(valid_values))
    fig.add_trace(
        go.Heatmap(
            z=np.flipud(overlay),
            opacity=opacity,
            colorscale=discrete_colorbar(colors),
            colorbar=colorbar,
            name="overlay",
        )
    )
    fig.add_layout_image(
        source=rgb_image,
        xref="x",
        yref="y",
        x=0.5,
        y=img_height - 0.5,
        sizex=img_width,
        sizey=img_height,
        sizing="stretch",
        layer="below",
    )

    scale = 1.5
    fig.layout.height = img_height * scale
    fig.layout.width = img_width * scale

    # Create and add slider
    steps = []
    opacity_range = np.arange(0, 1 + 0.05, 0.05)
    for o in opacity_range:
        steps.append({"method": "restyle", "args": [{"opacity": o}], "label": f"{o:.02f}"})

    opacity_slider = {
        "active": np.where(opacity_range == opacity)[0].item(),
        "currentvalue": {"prefix": "Opacity: "},
        "steps": steps,
    }

    fig.update_layout(sliders=[opacity_slider], title_x=0.5, title_text=path.image_name())
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))  # Keep the aspect ratio while zooming
    fig.layout.xaxis.showgrid = False
    fig.layout.yaxis.showgrid = False

    return fig


def prediction_figure_html(
    predictions: np.ndarray,
    confidence: np.ndarray,
    path: DataPath,
    config: Config,
    title_suffix: str = "",
    rgb_image: np.ndarray = None,
) -> str:
    """Create HTML which combines several visualizations for the network output. It shows the network predictions + confidence score and compares it to the reference segmentation if available.

    Note: The output is not a valid HTML document but just the HTML code from the Plotly figures. This is useful to combine the output of this function yet again with other plots.

    Args:
        predictions: Network label predictions (height, width).
        confidence: Network confidence scores (height, width).
        path: DataPath to the image.
        config: Configuration of the training run (used to extract the label mapping).
        title_suffix: Text which is added to the title of the figure.
        rgb_image: If not None, will show this image in the background instead of the reconstructed RGB image of the data path.

    Returns: HTML code of the figure.
    """
    assert predictions.shape == confidence.shape, "Predictions must have image dimensions"

    # Load original image
    if rgb_image is None:
        rgb_image = path.read_rgb_reconstructed()

    rgb_image = Image.fromarray(rgb_image)
    label_mapping = LabelMapping.from_config(config)
    img_height, img_width = path.dataset_settings["shape"][:2]

    reference = np.zeros(predictions.shape, np.float32)
    reference[:] = None
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Confidence", "Prediction"),
        vertical_spacing=0.05,
        horizontal_spacing=0.09,
    )

    def add_background_image(fig: go.Figure, row: int, col: int) -> None:
        fig.add_layout_image(
            source=rgb_image,
            xref="x",
            yref="y",
            x=0.5,
            y=img_height - 0.5,
            sizex=img_width,
            sizey=img_height,
            sizing="stretch",
            layer="below",
            row=row,
            col=col,
        )

    # Prediction plot
    def map_hover_prediction(pred, ref):
        if np.isnan(ref):
            return f"{label_mapping.index_to_name(pred)} (No reference available)"

        pred = int(pred)
        ref = int(ref)

        if pred == ref:
            return label_mapping.index_to_name(pred)
        else:
            return f"{label_mapping.index_to_name(pred)} instead of {label_mapping.index_to_name(ref)}"

    hover_text_predictions = np.vectorize(map_hover_prediction)(predictions, reference)

    # Color mapping based on the valid labels which occur in one of the two images (prediction or reference segmentation)
    available_label_indices = np.unique(
        np.concatenate([reference[~np.isnan(reference)], predictions[~np.isnan(predictions)]])
    )
    label_mapping_valid = {
        label_index: label_mapping.index_to_name(label_index)
        for label_index in label_mapping.label_indices()
        if label_index in available_label_indices
    }

    # We need to remap both images to consecutive values
    remapping_minimal = {label_index: i for i, label_index in enumerate(label_mapping_valid.keys())}
    max_label_index = len(remapping_minimal) - 1

    label_colors = [to_rgba(label_mapping.name_to_color(label)) for label in label_mapping_valid.values()]
    colorscale = discrete_colorbar(label_colors)
    colorbar = {
        "title": "class",
        "tickvals": list(range(len(label_mapping_valid))),
        "ticktext": list(label_mapping_valid.values()),
        "x": 1.0,
    }
    opacity = 0.5

    fig.add_trace(
        go.Heatmap(
            z=np.flipud(confidence),
            opacity=opacity,
            colorbar=dict(title="confidence score", titleside="right", x=0.45),
            name="Confidence",
        ),
        row=1,
        col=1,
    )
    add_background_image(fig, row=1, col=1)

    fig.add_trace(
        go.Heatmap(
            z=np.flipud(consecutive_segmentation(predictions, remapping_minimal)),
            zmin=0.0,
            zmax=max_label_index,
            opacity=opacity,
            colorscale=colorscale,
            colorbar=colorbar,
            text=np.flipud(hover_text_predictions),
            name="Prediction",
            showscale=True,
        ),
        row=1,
        col=2,
    )
    add_background_image(fig, row=1, col=2)

    fig.update_layout(
        title_x=0.5,
        title_text=f"{path.image_name()}{title_suffix}",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis2_showgrid=False,
        yaxis2_showgrid=False,
    )

    # Make all axes relative to the first (ensures that all images show the same excerpt) and ensure equal axes scaling (ensures that the image does not get distorted)
    height = img_height * 1.2
    width = img_width * 2
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    fig.update_layout(yaxis2=dict(scaleanchor="x2", scaleratio=1, matches="y"), xaxis2=dict(matches="x"))
    fig.update_layout(height=height, width=width, template="plotly_white")
    fig.update_yaxes(visible=False, showticklabels=False, row=1, col=2)

    # Create and add slider
    steps = []
    opacity_range = np.arange(0, 1 + 0.05, 0.05)
    for o in opacity_range:
        steps.append({"method": "restyle", "args": [{"opacity": o}], "label": f"{o:.02f}"})

    opacity_slider = {
        "active": np.where(opacity_range == opacity)[0].item(),
        "currentvalue": {"prefix": "Opacity: "},
        "steps": steps,
    }

    fig.update_layout(sliders=[opacity_slider])

    html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="prediction")

    # Load reference segmentation
    seg = path.read_segmentation(annotation_name="all")
    if seg is not None:
        # We combine the two plots via HTML and not with Plotly directly because it is hard to impossible with the different interactions of the segmentation figure (e.g. buttons for different annotators)
        fig_seg = create_segmentation_overlay(seg, path)
        fig_seg.update_layout(title_text="Reference", title_x=0.5)
        fig_seg.update_layout(height=height * 0.90, width=width * 0.70)
        html += fig_seg.to_html(full_html=False, include_plotlyjs="cdn", div_id="reference")

    return html


def create_image_scores_figure(labels: list[str], dice_classes: list[float]) -> go.Figure:
    assert len(labels) == len(dice_classes), "Labels and class dices must match"

    colors = [settings.label_colors[l] for l in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=dice_classes, marker=dict(color=colors)))
    fig.update_layout(title_x=0.5, title_text="Class dice scores")
    fig.layout.yaxis.title = "Dice metric"
    fig.layout.height = 300
    fig.layout.width = len(labels) * 120 + 100

    return fig


def create_ece_figure(df: pd.DataFrame) -> None:
    folds = sorted(df["fold_name"].unique())

    fig = go.Figure()
    # The buttons work by first drawing all plots and then switching of their visibility with the buttons
    button_states = []
    line_ids = []

    for f, fold_name in enumerate(folds):
        df_fold = df.query(f'fold_name == "{fold_name}"')
        df_fold = df_fold.query(f'epoch_index == {df_fold["best_epoch_index"].unique().item()}')

        # Aggregate information from all images
        acc_mat = np.stack([v["accuracies"] for v in df_fold["ece"]])
        conf_mat = np.stack([v["confidences"] for v in df_fold["ece"]])
        prob_mat = np.stack([v["probabilities"] for v in df_fold["ece"]])

        ece = ECE.aggregate_vectors(acc_mat, conf_mat, prob_mat)

        x = np.linspace(0, 1, len(ece["accuracies"]) + 1)

        hover_text = []
        for prob, conf in zip(ece["probabilities"], ece["confidences"]):
            hover_text.append(f"samples={prob:0.3f}, confidence={conf:0.3f}")

        fig.add_trace(
            go.Bar(
                x=x,
                y=ece["accuracies"],
                marker=dict(color=np.log(ece["probabilities"] + 0.001)),
                text=hover_text,
                name="Network",
                visible=f == 0,
            )
        )
        line_ids.append(f)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Optimum", visible=f == 0))
        line_ids.append(f)

        button_states.append({
            "label": fold_name,
            "method": "update",
            "args": [{"title": f'Confidence histogram ({fold_name}, ece={ece["error"]:0.3f})'}],
        })

    # Calculate the visible states (find out which lines have to be activated for which fold)
    line_ids = np.array(line_ids)
    for b, button_state in enumerate(button_states):
        visible_state = line_ids == b
        button_state["args"].insert(0, {"visible": visible_state.tolist()})

    fig.layout.update(updatemenus=[go.layout.Updatemenu(type="dropdown", active=0, buttons=button_states)])

    fig.layout.xaxis.title = "Confidence"
    fig.layout.yaxis.title = "Accuracy"
    fig.layout.title = button_states[0]["args"][1]["title"]
    fig.layout.width = 600
    fig.layout.height = 500

    return fig


def show_utilization(run_dir: Path) -> None:
    from htc.utils.helper_functions import load_util_log

    fig = go.Figure()
    button_states = []

    fold_dirs = sorted(run_dir.glob("fold*"))
    for f, fold_dir in enumerate(fold_dirs):
        data = load_util_log(fold_dir)

        time = np.array(data["raw_data"]["time"]) - data["raw_data"]["time"][0]
        time /= 3600

        fig.add_trace(go.Scatter(x=time, y=data["cpu_load_mean"], name="cpu_load", visible=f == 0))
        fig.add_trace(go.Scatter(x=time, y=data["gpu_load_mean"], name="gpu_load", visible=f == 0))

        visible_state = [False] * len(fold_dirs) * 2  # CPU and GPU plot
        visible_state[2 * f] = True
        visible_state[2 * f + 1] = True
        button_states.append({
            "label": fold_dir.stem,
            "method": "update",
            "args": [{"visible": visible_state}, {"title": f'avg_gpuutil = {np.mean(data["gpu_load_mean"]):0.4f}'}],
        })

    fig.layout.update(updatemenus=[go.layout.Updatemenu(type="dropdown", active=0, buttons=button_states)])

    fig.layout.xaxis.title = "time [h]"
    fig.layout.yaxis.title = "utilization"
    fig.layout.title = button_states[0]["args"][1]["title"]
    fig.update_layout(title_x=0.5)

    fig.show()


def create_training_stats_figure(run_dir: Path) -> go.Figure:
    data_specs = DataSpecification(run_dir / "data.json")

    # Compute matrix
    stats = []
    titles = []
    fold_dirs = sorted(run_dir.glob("fold*"))

    datasets = [{} for i in range(len(fold_dirs))]
    for i, fold_dir in enumerate(fold_dirs):
        train_stats = np.load(fold_dir / "trainings_stats.npz", allow_pickle=True)["data"]

        # Only used to get the number of training images for this fold, we don't need a mapping from image_index to image_name for this figure
        image_names = set()
        for name, paths in data_specs.folds[fold_dir.name].items():
            if name.startswith("train"):
                image_names.update([p.image_name() for p in paths])
                datasets[i][paths[0].dataset_settings["dataset_name"]] = len(
                    paths
                )  # as each dataset within the folds has the same

        assert len(image_names) > 0
        fold_stats = np.zeros((len(image_names), len(train_stats)), dtype=np.int64)

        for epoch_index, stats_epoch in enumerate(train_stats):
            for image_index in stats_epoch["img_indices"]:
                fold_stats[image_index, epoch_index] += 1

        stats.append(fold_stats)
        n_used_images = np.count_nonzero(np.sum(fold_stats, axis=1))
        titles.append(f"{fold_dir.name} ({n_used_images} of {len(image_names)} images used in total)")

    # cumsum of dataset counts for line drawing
    for i in range(len(datasets)):
        cumsum_vals = np.cumsum(list(datasets[i].values()))
        for j, (k, v) in enumerate(datasets[i].items()):
            datasets[i][k] = cumsum_vals[j]

    # Visualize
    fig = make_subplots(rows=len(stats), cols=1, subplot_titles=titles)

    for i, stat in enumerate(stats):
        fig.add_trace(go.Heatmap(z=stat, coloraxis="coloraxis"), row=i + 1, col=1)

        for dataset, count in datasets[i].items():
            fig.append_trace(
                go.Scatter(
                    x=[-0.5, stats[0].shape[1] - 0.5],
                    y=[count, count],
                    mode="lines",
                    line={"color": "red", "width": 2.0},
                ),
                row=i + 1,
                col=1,
            )
        fig.update_xaxes(title_text="epoch id", row=i + 1, col=1)
        fig.update_yaxes(title_text="img idx", row=i + 1, col=1)

    fig.update_layout(height=250 * len(fold_dirs), width=1000)
    fig.update_layout(coloraxis={"colorscale": "viridis"})
    fig.update_layout(showlegend=False)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis={"showgrid": False}, yaxis={"showgrid": False})

    return fig


def create_surface_dice_plot(
    dice_values: Union[list, np.ndarray], surface_values: Union[list, np.ndarray], **box_kwargs
) -> go.Figure:
    if len(dice_values) != len(surface_values):
        settings.log.warning(
            f"The number of values for dice ({dice_values}) is different to the number of surface values"
            f" ({surface_values})"
        )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Box(y=dice_values, name="dice", boxmean="sd", boxpoints="all", **box_kwargs), secondary_y=False)
    fig.add_trace(
        go.Box(y=surface_values, name="surface distance", boxmean="sd", boxpoints="all", **box_kwargs), secondary_y=True
    )
    fig.update_layout(
        title_x=0.5,
        title_text=(
            f"μ_dice={np.mean(dice_values):.03f} ± {np.std(dice_values):.03f},"
            f" μ_surface={np.mean(surface_values):.03f} ± {np.std(surface_values):.03f}"
        ),
    )
    fig.update_yaxes(title_text="Dice metric", secondary_y=False)
    fig.update_yaxes(title_text="Surface distance", secondary_y=True)

    return fig


def create_seed_comparison(base_dir: Path, name1: str, name2: str) -> go.Figure:
    runs1 = sorted(base_dir.glob(f"*seed=?,{name1}"))
    runs2 = sorted(base_dir.glob(f"*seed=?,{name2}"))
    assert len(runs1) == len(runs2)
    assert len(runs1) >= 2, f"At least two seed runs are required ({runs1 = }, {runs2 = })"

    # Calculate metrics
    metrics = ["dice_metric_image", "surface_distance_metric_image"]
    results1 = [MetricAggregation(r / "validation_table.pkl.xz").grouped_cm(additional_metrics=metrics) for r in runs1]
    results2 = [MetricAggregation(r / "validation_table.pkl.xz").grouped_cm(additional_metrics=metrics) for r in runs2]

    # Dice
    dices1 = [r["dice_metric_image"].mean() for r in results1]
    dices2 = [r["dice_metric_image"].mean() for r in results2]
    dices1_cm = [np.mean([dice_from_cm(row) for row in r["confusion_matrix"]]) for r in results1]
    dices2_cm = [np.mean([dice_from_cm(row) for row in r["confusion_matrix"]]) for r in results2]

    # Surface distance
    surfaces1 = [r["surface_distance_metric_image"].mean() for r in results1]
    surfaces2 = [r["surface_distance_metric_image"].mean() for r in results2]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            (
                f"μ_{name1}={np.mean(dices1):.03f} ± {np.std(dices1):.03f}<br>μ_{name2}={np.mean(dices2):.03f} ±"
                f" {np.std(dices2):.03f}<br>p={stats.ttest_ind(dices1, dices2).pvalue:.03f}"
            ),
            (
                f"μ_{name1}={np.mean(surfaces1):.03f} ±"
                f" {np.std(surfaces1):.03f}<br>μ_{name2}={np.mean(surfaces2):.03f} ±"
                f" {np.std(surfaces2):.03f}<br>p={stats.ttest_ind(surfaces1, surfaces2).pvalue:.03f}"
            ),
        ),
    )

    fig.add_trace(go.Box(y=dices1, boxpoints="all", boxmean="sd", name=name1, marker_color="indianred"), row=1, col=1)
    fig.add_trace(
        go.Box(y=dices2, boxpoints="all", boxmean="sd", name=name2, marker_color="lightseagreen"), row=1, col=1
    )
    fig.add_trace(
        go.Box(y=dices1_cm, boxpoints="all", boxmean="sd", name=f"cm_{name1}", marker_color="indianred"), row=1, col=1
    )
    fig.add_trace(
        go.Box(y=dices2_cm, boxpoints="all", boxmean="sd", name=f"cm_{name2}", marker_color="lightseagreen"),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Dice", row=1, col=1)

    fig.add_trace(
        go.Box(y=surfaces1, boxpoints="all", boxmean="sd", name=name1, marker_color="indianred"), row=2, col=1
    )
    fig.add_trace(
        go.Box(y=surfaces2, boxpoints="all", boxmean="sd", name=name2, marker_color="lightseagreen"), row=2, col=1
    )
    fig.update_yaxes(title_text="Average Surface Distance", row=2, col=1)

    fig.layout.height = 1000

    return fig


def add_std_fill(
    fig: go.Figure,
    mid_line: np.ndarray,
    std_range: np.ndarray,
    linecolor: str,
    label: str,
    row: int = None,
    col: int = None,
    x: Union[list, np.ndarray] = None,
    opacity: float = 0.15,
    **scatter_kwargs,
) -> go.Figure:
    assert mid_line.shape == std_range.shape, "Shapes of mid line and std range dont match!"

    if x is None:
        channels = list(np.arange(len(mid_line)))
    else:
        channels = list(x)

    upper_border = list(mid_line + std_range)
    lower_border = list(mid_line - std_range)
    lower_border = lower_border[::-1]

    if "hovertemplate" not in scatter_kwargs:
        scatter_kwargs["hovertemplate"] = (
            "wavelength: %{x:.1f}<br>normalized reflectance: %{y:.5f}<br>standard deviation: %{text:.5f}"
        )

    legendgroup = scatter_kwargs.pop("legendgroup", label)

    fig.add_trace(
        go.Scatter(
            x=channels,
            y=mid_line,
            text=std_range,
            name=label,
            line_color=linecolor,
            legendgroup=legendgroup,
            **scatter_kwargs,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=channels + channels[::-1],
            y=upper_border + lower_border,
            fill="toself",
            fillcolor=linecolor,
            line_color=linecolor,
            opacity=opacity,
            name=label,
            legendgroup=legendgroup,
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    fig.update_layout(hovermode="x")
    fig.update_xaxes(hoverformat=".1f")

    return fig


def colorchecker_fig_styling(fig: go.Figure, yaxis_title: str = "L1-normalized<br>reflectance [a.u.]") -> go.Figure:
    """This function takes a figure displaying colorchecker spectra in a grid of 6 rows and 4 columns and adds to each subplot a bar in the color of the corresponding colorchecker chip."""
    n_subplots = int(fig.data[-1]["xaxis"][1:])
    if n_subplots == 24:
        label_color_map = ColorcheckerReader.label_colors_classic
        width = 176
        y_offset = 0.006
        y_spacing = 0.086
        x_spacing = 0.255
        x_offset = 0.118
    elif n_subplots == 48:
        label_color_map = ColorcheckerReader.label_colors_passport
        width = 185
        y_offset = 0
        y_spacing = 0.085
        x_spacing = 0.2551
        x_offset = 0.118

    annotations = []
    for l, label in enumerate(label_color_map.keys()):
        if l in [1, 5, 6, 8, 10, 11, 15, 18, 19, 20, 24, 25, 26, 28, 29, 32, 33, 34, 35, 40, 41, 45, 46, 47]:
            font_color = "#000000"
        else:
            font_color = "#FFFFFF"
        annotations.append(
            dict(
                x=l % 4 * x_spacing + x_offset,  # horizontal alignment
                y=1 - (y_offset + l // 4 * y_spacing * 48 / n_subplots),  # vertical alignment
                xref="paper",
                yref="paper",
                text=label.replace("_", " "),
                bgcolor=label_color_map[label],
                font_color=font_color,
                width=width,
                showarrow=False,
            )
        )

    fig.update_xaxes(tickvals=[600, 700, 800, 900], ticktext=["600", "700", "800", "900"], tickmode="array")
    fig.update_yaxes(dtick=0.005)

    layout = {}
    for i in np.arange(n_subplots - 3, n_subplots + 1):
        layout[f"xaxis{i}"] = dict(title="wavelength [nm]")
    for i in np.arange(5, n_subplots, 4):
        layout[f"yaxis{i}"] = dict(title=yaxis_title)
    layout["yaxis"] = dict(title=yaxis_title)
    fig.update_layout(**layout)

    fig.update_layout(
        height=820 * n_subplots // 24,
        width=850,
        template="plotly_white",
        font_family="Arial",
        margin=dict(l=0, r=5, b=50, t=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=15)),
    )
    fig.update_layout(annotations=annotations)

    return fig


def create_spec_labels_figure(config: Config) -> go.Figure:
    spec = DataSpecification.from_config(config)
    mapping = LabelMapping.from_config(config)

    subject_matrix = np.zeros((len(spec), len(mapping)), dtype=np.int64)
    image_matrix = np.zeros((len(spec), len(mapping)), dtype=np.int64)

    for fold_index, fold_name in enumerate(spec.fold_names()):
        df = basic_statistics(paths=spec.fold_paths(fold_name, "train"), label_mapping=mapping)

        subject_stats = df.groupby(["label_index"], as_index=False)["subject_name"].agg("nunique")
        for _, row in subject_stats.iterrows():
            subject_matrix[fold_index, row["label_index"]] = row["subject_name"]

        image_stats = df.groupby(["label_index"], as_index=False)["image_name"].agg("nunique")
        for _, row in image_stats.iterrows():
            image_matrix[fold_index, row["label_index"]] = row["image_name"]

    fig = make_subplots(rows=2, cols=1, subplot_titles=["subjects", "images"], shared_xaxes=True, vertical_spacing=0.09)
    common_settings = {
        "x": mapping.label_names(),
        "y": spec.fold_names(),
        "texttemplate": "%{text}",
        "colorscale": "Teal",
    }
    fig.add_trace(
        go.Heatmap(
            z=subject_matrix,
            text=subject_matrix,
            hovertemplate="label: %{x}<br>fold: %{y}<br># subjects: %{z}",
            colorbar_len=0.5,
            colorbar_y=0.77,
            colorbar_title="# subjects",
            **common_settings,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=image_matrix,
            text=image_matrix,
            hovertemplate="label: %{x}<br>fold: %{y}<br># images: %{z}",
            colorbar_len=0.5,
            colorbar_y=0.23,
            colorbar_title="# images",
            **common_settings,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(title_x=0.5, title_text="Number of unique subjects and images per label in the training sets")
    fig.update_layout(height=600, width=800, margin=dict(l=0, r=0, b=0, t=80))

    return fig


def create_training_stats_label_figure(run_dir: Path) -> go.Figure:
    config = Config(run_dir / "config.json")
    spec = DataSpecification.from_config(config)
    mapping = LabelMapping.from_config(config)

    image_matrix = np.zeros((len(spec), len(mapping)), dtype=np.int64)

    for fold_index, fold_name in enumerate(spec.fold_names()):
        train_stats = np.load(run_dir / fold_name / "trainings_stats.npz", allow_pickle=True)["data"]
        paths = spec.fold_paths(fold_name, "train")
        image_names = [p.image_name() for p in paths]

        task = Task.from_config(config)
        if task == Task.SEGMENTATION:
            label_column = "label_index_mapped"
        elif task == Task.CLASSIFICATION:
            label_column = "image_labels"
        else:
            raise ValueError(f"Unknown task: {task}")

        # We are using the median table to retrieve the label indices for each image
        df = median_table(paths=spec.fold_paths(fold_name, "train"), config=config)

        # There may be more than one entry for one image and one label due to the label mapping
        # We are not interested in the duplicates here because each target label (as defined by the label mapping) counts only once per image
        df = df[["image_name", label_column]].drop_duplicates()

        # The training stats only store the index of the image in the training set so we need to add this information to the table (based on the specs which is also used during training)
        df["image_index"] = [image_names.index(n) for n in df["image_name"]]
        df.set_index("image_index", inplace=True)

        for stats_epoch in train_stats:
            # An image appears multiple times in the selection table if the same index appears more than once
            df_epoch = df.loc[stats_epoch["img_indices"]]

            image_stats = df_epoch.groupby([label_column], as_index=False)["image_name"].agg("count")
            for _, row in image_stats.iterrows():
                image_matrix[fold_index, row[label_column]] += row["image_name"]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=image_matrix,
            x=mapping.label_names(),
            y=spec.fold_names(),
            text=image_matrix,
            texttemplate="%{text}",
            hovertemplate="label: %{x}<br>fold: %{y}<br># images: %{z}",
            colorscale="Teal",
            colorbar_title="# images",
        )
    )

    fig.update_layout(title_x=0.5, title_text="Total number of images seen per label during training")
    fig.update_layout(height=400, width=len(mapping) * 50 + 200, margin=dict(l=0, r=0, b=0, t=40))

    return fig
