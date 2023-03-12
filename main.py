import requests
import urllib.request
from bs4 import BeautifulSoup as BS
import re


def qiangke1():
    url = "http://jiaowu.hncst.edu.cn/xf_xsqxxxk.aspx?xh={0}".format(number)
    header = {
        "Referer": "http://jiaowu.hncst.edu.cn/xs_main.aspx?xh={0}".format(number),
        "Cookie": "ASP.NET_SessionId={0}".format(cookie),
    }
    for x in range(1, 100):
        response = requests.get(url, headers=header)
        if len(response.text) > 5000:
            soup = BS(response.text, 'html.parser')
            # 从页面内容中提取表格内容
            table_pattern = re.compile('<table.*?id="DataGrid2".*?>(.*?)</table>', re.S)
            cell_pattern = re.compile('<td.*?>(.*?)</td>', re.S)
            table_html = str(soup.find("table", id="DataGrid2"))
            #print(table_html)
            print(len(table_html))
            table_match = table_pattern.search(table_html)
            if table_match:
                table_text = table_match.group(1)
                rows = table_text.split('</tr>')
                table_data = []
                for row in rows:
                    cells = cell_pattern.findall(row)
                    table_data.append(cells)

                # 打印表格
                for row in table_data:
                    row_text = [re.sub('<[^>]+>', '', cell) for cell in row]  # 使用正则表达式去掉HTML标签
                    print('| ' + ' | '.join(row_text) + ' |')  # 打印去掉HTML标签后的表格
            break

def qiangke2():
    data = {
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "__VIEWSTATE": "dDwtMTUzMzc4ODUxMDt0PHA8bDxzb3J0T3JkZXI7enlkbTt4eTtkcXN6ajt6eW1jO3h6YjtYS1hRO3htO1hLWE47PjtsPCBhc2MgOzAxMDE76L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJOzIwMjE76L2v5Lu25oqA5pyv77yIM++8iTsyMei9r+S7tuaKgOacrzMwMTsyO+mSn+aWh+aZujsyMDIyLTIwMjM7Pj47bDxpPDE+Oz47bDx0PDtsPGk8MT47aTwzPjtpPDU+O2k8Nz47aTw5PjtpPDE0PjtpPDIyPjtpPDIzPjtpPDI3PjtpPDMxPjtpPDMzPjs+O2w8dDxwPHA8bDxUZXh0Oz47bDzlp5PlkI3vvJrpkp/mlofmmbombmJzcFw7Jm5ic3BcOyZuYnNwXDsmbmJzcFw75a2m6Zmi77ya6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJJm5ic3BcOyZuYnNwXDsmbmJzcFw7Jm5ic3BcO+S4k+S4mu+8mui9r+S7tuaKgOacr++8iDPvvIk7Pj47Pjs7Pjt0PHQ8cDxwPGw8RGF0YVRleHRGaWVsZDtEYXRhVmFsdWVGaWVsZDs+O2w8a2N4ejtrY3h6Oz4+Oz47dDxpPDI+O0A85YWs5YWx6YCJ5L+u6K++O1xlOz47QDzlhazlhbHpgInkv67or747XGU7Pj47bDxpPDE+Oz4+Ozs+O3Q8dDw7cDxsPGk8MD47aTwxPjtpPDI+Oz47bDxwPOaciTvmnIk+O3A85pegO+aXoD47cDxcZTtcZT47Pj47bDxpPDA+Oz4+Ozs+O3Q8dDxwPHA8bDxEYXRhVGV4dEZpZWxkO0RhdGFWYWx1ZUZpZWxkOz47bDxrY2dzO2tjZ3M7Pj47Pjt0PGk8ND47QDzkurrmloflrabnp5FB57G76YCJ5L+uO+S6uuaWh+WtpuenkULnsbvpgInkv6476Ieq54S256eR5a2m5bel56iL5oqA5pyv6YCJ5L+uO1xlOz47QDzkurrmloflrabnp5FB57G76YCJ5L+uO+S6uuaWh+WtpuenkULnsbvpgInkv6476Ieq54S256eR5a2m5bel56iL5oqA5pyv6YCJ5L+uO1xlOz4+O2w8aTwzPjs+Pjs7Pjt0PHQ8cDxwPGw8RGF0YVRleHRGaWVsZDtEYXRhVmFsdWVGaWVsZDs+O2w8eHFtYzt4cWRtOz4+Oz47dDxpPDE+O0A85pys6YOoOz47QDwxOz4+O2w8aTwwPjs+Pjs7Pjt0PHQ8cDxwPGw8RGF0YVRleHRGaWVsZDtEYXRhVmFsdWVGaWVsZDs+O2w8c2tzajtza3NqOz4+Oz47dDxpPDc+O0A85ZGo5LqM56ysNyw46IqCe+esrDUtMTjlkah9O+WRqOS4ieesrDcsOOiKgnvnrKw1LTE45ZGofTvlkajkuInnrKw5LDEw6IqCe+esrDUtMTjlkah9O+WRqOWbm+esrDcsOOiKgnvnrKw1LTE45ZGofTvlkajlm5vnrKw5LDEw6IqCe+esrDUtMTjlkah9O+WRqOS4gOesrDcsOOiKgnvnrKw1LTE45ZGofTtcZTs+O0A85ZGo5LqM56ysNyw46IqCe+esrDUtMTjlkah9O+WRqOS4ieesrDcsOOiKgnvnrKw1LTE45ZGofTvlkajkuInnrKw5LDEw6IqCe+esrDUtMTjlkah9O+WRqOWbm+esrDcsOOiKgnvnrKw1LTE45ZGofTvlkajlm5vnrKw5LDEw6IqCe+esrDUtMTjlkah9O+WRqOS4gOesrDcsOOiKgnvnrKw1LTE45ZGofTtcZTs+PjtsPGk8Nj47Pj47Oz47dDxAMDxwPHA8bDxQYWdlQ291bnQ7XyFJdGVtQ291bnQ7XyFEYXRhU291cmNlSXRlbUNvdW50O0RhdGFLZXlzOz47bDxpPDE+O2k8MT47aTwxPjtsPD47Pj47Pjs7Ozs7Ozs7Ozs+O2w8aTwwPjs+O2w8dDw7bDxpPDE+Oz47bDx0PDtsPGk8MT47aTwyPjtpPDM+O2k8ND47aTw1PjtpPDY+O2k8Nz47aTw4PjtpPDk+O2k8MTA+O2k8MTE+O2k8MTI+O2k8MTM+O2k8MTQ+O2k8MTU+O2k8MTY+O2k8MTc+O2k8MTg+O2k8MTk+O2k8MjA+O2k8MjE+O2k8MjI+O2k8MjM+O2k8MjU+O2k8MjY+O2k8Mjc+O2k8Mjg+Oz47bDx0PDtsPGk8Mz47PjtsPHQ8cDxsPG9uY2xpY2s7VmlzaWJsZTs+O2w8c2hvdyh0aGlzLCd8fHwnKTtvPGY+Oz4+Ozs+Oz4+O3Q8cDxwPGw8VGV4dDs+O2w8XDxhIGhyZWY9JyMnIG9uY2xpY2s9IndpbmRvdy5vcGVuKCdrY3h4LmFzcHg/eGg9MjAyMTAxMzI0NiZrY2RtPVNLWDAzMzM0Jnhra2g9KDIwMjItMjAyMy0yKS1TS1gwMzMzNC05MDE2OC0xJywna2N4eCcsJ3Rvb2xiYXI9MCxsb2NhdGlvbj0wLGRpcmVjdG9yaWVzPTAsc3RhdHVzPTAsbWVudWJhcj0wLHNjcm9sbGJhcnM9MSxyZXNpemFibGU9MCx3aWR0aD01NDAsaGVpZ2h0PTQ1MCxsZWZ0PTEyMCx0b3A9NjAnKSJcPuS/oeaBr+ajgOe0ou+8iOe9kee7nOivvueoi++8iVw8L2FcPjs+Pjs+Ozs+O3Q8O2w8aTwxPjs+O2w8dDxwPHA8bDxUZXh0Oz47bDzmnKrkuIrkvKA7Pj47Pjs7Pjs+Pjt0PHA8cDxsPFRleHQ7PjtsPFNLWDAzMzM0Oz4+Oz47Oz47dDxwPHA8bDxUZXh0Oz47bDxcPGEgaHJlZj0nIycgb25jbGljaz0id2luZG93Lm9wZW4oJ2pzeHguYXNweD94aD0yMDIxMDEzMjQ2JmpzemdoPTkwMTY4Jnhra2g9KDIwMjItMjAyMy0yKS1TS1gwMzMzNC05MDE2OC0xJywnanN4eCcsJ3Rvb2xiYXI9MCxsb2NhdGlvbj0wLGRpcmVjdG9yaWVzPTAsc3RhdHVzPTAsbWVudWJhcj0wLHNjcm9sbGJhcnM9MSxyZXNpemFibGU9MCx3aWR0aD01NDAsaGVpZ2h0PTQ1MCxsZWZ0PTEyMCx0b3A9NjAnKSJcPuWQtOS5vua4hVw8L2FcPjs+Pjs+Ozs+O3Q8cDxwPGw8VGV4dDtUb29sVGlwOz47bDzlkajkuInnrKw5LDEw6IqCe+esrC4uLjvlkajkuInnrKw5LDEw6IqCe+esrDUtMTjlkah9Oz4+Oz47Oz47dDxwPHA8bDxUZXh0Oz47bDznlLXlrZDpmIXop4jlrqQ7Pj47Pjs7Pjt0PHA8cDxsPFRleHQ7PjtsPDIuMDs+Pjs+Ozs+O3Q8cDxwPGw8VGV4dDs+O2w8Mi4wLTAuMDs+Pjs+Ozs+O3Q8cDxwPGw8VGV4dDs+O2w8MDUtMTg7Pj47Pjs7Pjt0PHA8cDxsPFRleHQ7PjtsPDQ1Oz4+Oz47Oz47dDxwPHA8bDxUZXh0Oz47bDw0NDs+Pjs+Ozs+O3Q8cDxwPGw8VGV4dDs+O2w8MTs+Pjs+Ozs+O3Q8cDxwPGw8VGV4dDs+O2w8KDIwMjItMjAyMy0yKS1TS1gwMzMzNC05MDE2OC0xOz4+Oz47Oz47dDxwPHA8bDxUZXh0Oz47bDwmbmJzcFw7Oz4+Oz47Oz47dDxwPHA8bDxUZXh0Oz47bDxTS1gwMzMzNDs+Pjs+Ozs+O3Q8cDxwPGw8VGV4dDs+O2w8OTAxNjg7Pj47Pjs7Pjt0PHA8cDxsPFRleHQ7PjtsPOS6uuaWh+WtpuenkULnsbvpgInkv647Pj47Pjs7Pjt0PHA8cDxsPFRleHQ7PjtsPOWFrOWFsemAieS/ruivvjs+Pjs+Ozs+O3Q8cDxwPGw8VGV4dDs+O2w85pys6YOoOz4+Oz47Oz47dDxwPHA8bDxUZXh0Oz47bDwmbmJzcFw7Oz4+Oz47Oz47dDxwPHA8bDxUZXh0Oz47bDzmlZnliqHlpIQ7Pj47Pjs7Pjt0PHA8cDxsPFRleHQ7PjtsPCZuYnNwXDs7Pj47Pjs7Pjt0PHA8cDxsPFRleHQ7PjtsPCZuYnNwXDs7Pj47Pjs7Pjt0PHA8cDxsPFRleHQ7PjtsPCZuYnNwXDs7Pj47Pjs7Pjt0PHA8cDxsPFRleHQ7PjtsPCZuYnNwXDs7Pj47Pjs7Pjt0PHA8cDxsPFRleHQ7PjtsPCZuYnNwXDs7Pj47Pjs7Pjs+Pjs+Pjs+Pjt0PHA8cDxsPFF1ZXJ5O2R0UmVjb3Jkczs+O2w8c2VsZWN0ICogZnJvbSAoc2VsZWN0IGEua2NkbSxhLmtjbWMsYS5qc3pnaCxhLmpzeG0sYS5za3NqLGEuc2tkZCxhLnhmLGEuenhzLGEucXNqc3osYS54a2toLGEueHFicyxhLmtjZ3MsYS5rY3h6LG52bChhLnJzLDApIHJzLChzZWxlY3QgY291bnQoeGgpIGZyb20geHN4a2JuIGYgd2hlcmUgYS54a2toPWYueGtraCkgICB5eHJzLCcnIHlsLGEuYnosYS5ta3poICxhLmtreHksYS5rc3NqLCcnIHNmYngsYS5rc3hzLGEuc3FzbSAsIGNhc2Ugd2hlbiBiLmZqZHogaXMgbnVsbCB0aGVuICcnIGVsc2UgIGIuZmpkeiBlbmQgZmp4eiAsIGpjbWN8fCd8J3x8Y2JzfHwnfCd8fHp6fHwnfCd8fGJiIGpjbnIgZnJvbSAgeHhranhyd2IgIGEgICxqc2p4cmxiX2ZqZHogYiAgd2hlcmUgYS54bj0nMjAyMi0yMDIzJyBhbmQgYS54cT0nMicgYW5kICAgYS54a2toIGxpa2UgJygyMDIyLTIwMjMtMiktJScgYW5kIGEueGtraD1iLnhra2goKykgIGFuZCBhLnhra2ggbGlrZSAnKDIwMjItMjAyMy0yKS0lJyBhbmQgYS54a3p0PScxJyAgIGFuZCBleGlzdHMgKHNlbGVjdCAneCcgZnJvbSAoIHNlbGVjdCAgYS54a21jLGEuYmggIGZyb20gIHhrZG1iIGEgLHp5ZG1iIGIgd2hlcmUgYi56eWRtPScwMTAxJyBhbmQgYi5reGxiIGxpa2UgJyUnfHxhLmJofHwnJScpeCB3aGVyZSBhLmtjZ3M9eC54a21jKSBhbmQgYS54cWJzPSfmnKzpg6gnICBhbmQgKG14ZHggaXMgbnVsbCBvciBteGR4IGxpa2UgJyUsJ3x8JzIwMjEwMTMyNDYnfHwnLCUnICAgICAgICAgICAgb3IgbXhkeCBsaWtlICclLCd8fCfova/ku7bmioDmnK/vvIgz77yJSkFWQUVFJ3x8JywlJyAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwnMjAyMee6p+i9r+S7tuaKgOacr++8iDPvvIlKQVZBRUUnfHwnLCUnICAgIG9yIG14ZHggbGlrZSAnJSwnfHwn5LiT56eRJ3x8JywlJyAgICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwn5LiT56eRMjAyMee6pyd8fCcsJScgICAgICAgICAgb3IgbXhkeCBsaWtlICclLCd8fCfkuJPnp5HnlLfnlJ8nfHwnLCUnICAgICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwn5LiT56eR6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJJ3x8JywlJyAgICAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8J+S4k+enkTIwMjHnuqfova/ku7blt6XnqIvlrabpmaLvvIjkurrlt6Xmmbrog73lrabpmaLvvIknfHwnLCUnICAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8J+S4k+enkTIwMjHnuqfnlLfnlJ8nfHwnLCUnICAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8J+S4k+enkei9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8ieeUt+eUnyd8fCcsJScgICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwn5LiT56eRMjAyMee6p+i9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8ieeUt+eUnyd8fCcsJScgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8JzIwMjHnuqcnfHwnLCUnICAgICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwn6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJJ3x8JywlJyAgICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwn6L2v5Lu25oqA5pyv77yIM++8iSd8fCcsJScgICAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8JzIx6L2v5Lu25oqA5pyvMzAxJ3x8JywlJyAgICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwn55S355SfJ3x8JywlJyAgICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwnMjAyMee6p+i9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8iSd8fCcsJScgICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwnMjAyMee6p+i9r+S7tuaKgOacr++8iDPvvIknfHwnLCUnICAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8JzIwMjHnuqfnlLfnlJ8nfHwnLCUnICAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8J+i9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8ieeUt+eUnyd8fCcsJScgICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwn6L2v5Lu25oqA5pyv77yIM++8ieeUt+eUnyd8fCcsJScgICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwnMjHova/ku7bmioDmnK8zMDHnlLfnlJ8nfHwnLCUnICAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8JzIwMjHnuqfova/ku7blt6XnqIvlrabpmaLvvIjkurrlt6Xmmbrog73lrabpmaLvvInnlLfnlJ8nfHwnLCUnICAgICAgb3IgbXhkeCBsaWtlICclLCd8fCcyMDIx57qn6L2v5Lu25oqA5pyv77yIM++8ieeUt+eUnyd8fCcsJScgICAgb3IgbXhkeCBsaWtlICclLCd8fCcyMDIx57qnJ3x8JywlJyAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8J+i9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8iSd8fCcsJScgICAgICAgb3IgbXhkeCBsaWtlICclLCd8fCfova/ku7bmioDmnK/vvIgz77yJJ3x8JywlJyAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8JzIx6L2v5Lu25oqA5pyvMzAxJ3x8JywlJyAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8JzIwMjHnuqfova/ku7blt6XnqIvlrabpmaLvvIjkurrlt6Xmmbrog73lrabpmaLvvIknfHwnLCUnICAgICBvciBteGR4IGxpa2UgJyUsJ3x8JzIwMjHnuqfova/ku7bmioDmnK/vvIgz77yJJ3x8JywlJyAgIG9yIG14ZHggbGlrZSAnJSwnfHwnJ3x8JywlJyAgICAgb3IgbXhkeCBsaWtlICclLCd8fCfkuJPnp5EnfHwnLCUnICAgICAgICAgb3IgbXhkeCBsaWtlICclLCd8fCfkuJPnp5EyMDIx57qnJ3x8JywlJyAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8J+S4k+enkei9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8iSd8fCcsJScgICAgICAgb3IgbXhkeCBsaWtlICclLCd8fCfkuJPnp5EyMDIx57qn6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJJ3x8JywlJyAgICAgb3IgbXhkeCBsaWtlICclLCd8fCcyMDIx57qn55S355SfJ3x8JywlJyAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8J+i9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8ieeUt+eUnyd8fCcsJScgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwn6L2v5Lu25oqA5pyv77yIM++8ieeUt+eUnyd8fCcsJScgICBvciBteGR4IGxpa2UgJyUsJ3x8JzIx6L2v5Lu25oqA5pyvMzAx55S355SfJ3x8JywlJyAgICAgb3IgbXhkeCBsaWtlICclLCd8fCcyMDIx57qn6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJ55S355SfJ3x8JywlJyAgICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwnMjAyMee6p+i9r+S7tuaKgOacr++8iDPvvInnlLfnlJ8nfHwnLCUnICAgICAgIG9yIG14ZHggbGlrZSAnJSwnfHwn5LiT56eR55S355SfJ3x8JywlJyAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8J+S4k+enkTIwMjHnuqfnlLfnlJ8nfHwnLCUnICAgICBvciBteGR4IGxpa2UgJyUsJ3x8J+S4k+enkei9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8ieeUt+eUnyd8fCcsJScgICAgICAgb3IgbXhkeCBsaWtlICclLCd8fCfkuJPnp5EyMDIx57qn6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJ55S355SfJ3x8JywlJyAgICAgb3IgbXhkeCBsaWtlICclLCd8fCfova/ku7bmioDmnK/vvIgz77yJSkFWQUVFJ3x8JywlJyAgICAgICBvciBteGR4IGxpa2UgJyUsJ3x8JzIwMjHnuqfova/ku7bmioDmnK/vvIgz77yJSkFWQUVFJ3x8JywlJyAgICBvciBteGR4IGxpa2UgJyUsJ3x8Jyd8fCcsJScgICAgICAgb3IgbXhkeCBsaWtlICclLCd8fCcyMDIx57qnJ3x8JywlJyAgICBvciBteGR4IGxpa2UgJyUsJ3x8J+i9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8iSd8fCcsJScgICAgICApICAgICBhbmQgKHh6ZHggaXMgbnVsbCBvciB4emR4IG5vdCBsaWtlICclLCd8fCcyMDIxMDEzMjQ2J3x8JywlJyAgICAgICAgYW5kIHh6ZHggbm90IGxpa2UgJyUsJ3x8J+i9r+S7tuaKgOacr++8iDPvvIlKQVZBRUUnfHwnLCUnICBhbmQgeHpkeCBub3QgbGlrZSAnJSwnfHwnMjAyMee6p+i9r+S7tuaKgOacr++8iDPvvIlKQVZBRUUnfHwnLCUnIGFuZCB4emR4IG5vdCBsaWtlICclLCd8fCfkuJPnp5EnfHwnLCUnICAgICBhbmQgeHpkeCBub3QgbGlrZSAnJSwnfHwn5LiT56eRMjAyMee6pyd8fCcsJScgICAgICBhbmQgeHpkeCBub3QgbGlrZSAnJSwnfHwn5LiT56eR55S355SfJ3x8JywlJyAgICAgIGFuZCB4emR4IG5vdCBsaWtlICclLCd8fCfkuJPnp5Hova/ku7blt6XnqIvlrabpmaLvvIjkurrlt6Xmmbrog73lrabpmaLvvIknfHwnLCUnICAgICAgYW5kIHh6ZHggbm90IGxpa2UgJyUsJ3x8J+S4k+enkTIwMjHnuqfova/ku7blt6XnqIvlrabpmaLvvIjkurrlt6Xmmbrog73lrabpmaLvvIknfHwnLCUnICAgIGFuZCB4emR4IG5vdCBsaWtlICclLCd8fCfkuJPnp5EyMDIx57qn55S355SfJ3x8JywlJyAgICBhbmQgeHpkeCBub3QgbGlrZSAnJSwnfHwn5LiT56eR6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJ55S355SfJ3x8JywlJyAgICBhbmQgeHpkeCBub3QgbGlrZSAnJSwnfHwn5LiT56eRMjAyMee6p+i9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8ieeUt+eUnyd8fCcsJScgIGFuZCB4emR4IG5vdCBsaWtlICclLCd8fCcyMDIx57qnJ3x8JywlJyAgICAgIGFuZCB4emR4IG5vdCBsaWtlICclLCd8fCfova/ku7blt6XnqIvlrabpmaLvvIjkurrlt6Xmmbrog73lrabpmaLvvIknfHwnLCUnICAgICBhbmQgeHpkeCBub3QgbGlrZSAnJSwnfHwn6L2v5Lu25oqA5pyv77yIM++8iSd8fCcsJScgICAgIGFuZCB4emR4IG5vdCBsaWtlICclLCd8fCcyMei9r+S7tuaKgOacrzMwMSd8fCcsJScgICAgIGFuZCB4emR4IG5vdCBsaWtlICclLCd8fCfnlLfnlJ8nfHwnLCUnICAgICBhbmQgeHpkeCBub3QgbGlrZSAnJSwnfHwnMjAyMee6p+i9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8iSd8fCcsJScgICAgYW5kIHh6ZHggbm90IGxpa2UgJyUsJ3x8JzIwMjHnuqfova/ku7bmioDmnK/vvIgz77yJJ3x8JywlJyAgICBhbmQgeHpkeCBub3QgbGlrZSAnJSwnfHwnMjAyMee6p+eUt+eUnyd8fCcsJScgICAgYW5kIHh6ZHggbm90IGxpa2UgJyUsJ3x8J+i9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8ieeUt+eUnyd8fCcsJScgICAgYW5kIHh6ZHggbm90IGxpa2UgJyUsJ3x8J+i9r+S7tuaKgOacr++8iDPvvInnlLfnlJ8nfHwnLCUnICAgIGFuZCB4emR4IG5vdCBsaWtlICclLCd8fCcyMei9r+S7tuaKgOacrzMwMeeUt+eUnyd8fCcsJScgICAgYW5kIHh6ZHggbm90IGxpa2UgJyUsJ3x8JzIwMjHnuqfova/ku7blt6XnqIvlrabpmaLvvIjkurrlt6Xmmbrog73lrabpmaLvvInnlLfnlJ8nfHwnLCUnICBhbmQgeHpkeCBub3QgbGlrZSAnJSwnfHwnMjAyMee6p+i9r+S7tuaKgOacr++8iDPvvInnlLfnlJ8nfHwnLCUnICBhbmQgeHpkeCBub3QgIGxpa2UgJyUsJ3x8JzIwMjHnuqcnfHwnLCUnICAgICAgIGFuZCB4emR4IG5vdCAgbGlrZSAnJSwnfHwn6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJJ3x8JywlJyAgICAgICBhbmQgeHpkeCBub3QgIGxpa2UgJyUsJ3x8J+i9r+S7tuaKgOacr++8iDPvvIknfHwnLCUnICAgICAgIGFuZCB4emR4IG5vdCAgbGlrZSAnJSwnfHwnMjHova/ku7bmioDmnK8zMDEnfHwnLCUnICAgICAgIGFuZCB4emR4IG5vdCAgbGlrZSAnJSwnfHwnMjAyMee6p+i9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8iSd8fCcsJScgICAgIGFuZCB4emR4IG5vdCAgbGlrZSAnJSwnfHwnMjAyMee6p+i9r+S7tuaKgOacr++8iDPvvIknfHwnLCUnICAgIGFuZCB4emR4IG5vdCAgbGlrZSAnJSwnfHwnJ3x8JywlJyAgICAgYW5kIHh6ZHggbm90ICBsaWtlICclLCd8fCfkuJPnp5EnfHwnLCUnICAgICAgICAgYW5kIHh6ZHggbm90ICBsaWtlICclLCd8fCfkuJPnp5EyMDIx57qnJ3x8JywlJyAgICAgICBhbmQgeHpkeCBub3QgIGxpa2UgJyUsJ3x8J+S4k+enkei9r+S7tuW3peeoi+WtpumZou+8iOS6uuW3peaZuuiDveWtpumZou+8iSd8fCcsJScgICAgICAgYW5kIHh6ZHggbm90ICBsaWtlICclLCd8fCfkuJPnp5EyMDIx57qn6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJJ3x8JywlJyAgICAgYW5kIHh6ZHggbm90ICAgbGlrZSAnJSwnfHwnMjAyMee6p+eUt+eUnyd8fCcsJScgICAgICAgYW5kIHh6ZHggbm90ICAgbGlrZSAnJSwnfHwn6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJ55S355SfJ3x8JywlJyAgICAgYW5kIHh6ZHggbm90ICAgbGlrZSAnJSwnfHwn6L2v5Lu25oqA5pyv77yIM++8ieeUt+eUnyd8fCcsJScgICAgYW5kIHh6ZHggbm90ICAgbGlrZSAnJSwnfHwnMjHova/ku7bmioDmnK8zMDHnlLfnlJ8nfHwnLCUnICAgICBhbmQgeHpkeCBub3QgICBsaWtlICclLCd8fCcyMDIx57qn6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJ55S355SfJ3x8JywlJyAgICAgICAgIGFuZCB4emR4IG5vdCAgIGxpa2UgJyUsJ3x8JzIwMjHnuqfova/ku7bmioDmnK/vvIgz77yJ55S355SfJ3x8JywlJyAgICAgICBhbmQgeHpkeCBub3QgICBsaWtlICclLCd8fCfkuJPnp5HnlLfnlJ8nfHwnLCUnICAgICAgIGFuZCB4emR4IG5vdCAgIGxpa2UgJyUsJ3x8J+S4k+enkTIwMjHnuqfnlLfnlJ8nfHwnLCUnICAgICBhbmQgeHpkeCBub3QgICBsaWtlICclLCd8fCfkuJPnp5Hova/ku7blt6XnqIvlrabpmaLvvIjkurrlt6Xmmbrog73lrabpmaLvvInnlLfnlJ8nfHwnLCUnICAgICAgIGFuZCB4emR4IG5vdCAgIGxpa2UgJyUsJ3x8J+S4k+enkTIwMjHnuqfova/ku7blt6XnqIvlrabpmaLvvIjkurrlt6Xmmbrog73lrabpmaLvvInnlLfnlJ8nfHwnLCUnICAgICBhbmQgeHpkeCBub3QgbGlrZSAnJSwnfHwn6L2v5Lu25oqA5pyv77yIM++8iUpBVkFFRSd8fCcsJScgICAgICAgYW5kIHh6ZHggbm90ICBsaWtlICclLCd8fCcyMDIx57qn6L2v5Lu25oqA5pyv77yIM++8iUpBVkFFRSd8fCcsJScgICAgIGFuZCB4emR4IG5vdCBsaWtlICclLCd8fCcnfHwnLCUnICAgICAgIGFuZCB4emR4IG5vdCAgbGlrZSAnJSwnfHwnMjAyMee6pyd8fCcsJScgICAgIGFuZCB4emR4IG5vdCAgbGlrZSAnJSwnfHwn6L2v5Lu25bel56iL5a2m6Zmi77yI5Lq65bel5pm66IO95a2m6Zmi77yJJ3x8JywlJyAgICAgKSAgICBhbmQgc2Z4Z3hrPSfmmK8nKSB3aGVyZSAocnMteXhycylcPjAgb3JkZXIgYnkga2NkbSxqc3pnaDtiPEFBRUFBQUQvLy8vL0FRQUFBQUFBQUFBTUFnQUFBRkZUZVhOMFpXMHVSR0YwWVN3Z1ZtVnljMmx2YmoweExqQXVOVEF3TUM0d0xDQkRkV3gwZFhKbFBXNWxkWFJ5WVd3c0lGQjFZbXhwWTB0bGVWUnZhMlZ1UFdJM04yRTFZelUyTVRrek5HVXdPRGtGQVFBQUFCVlRlWE4wWlcwdVJHRjBZUzVFWVhSaFZHRmliR1VDQUFBQUNWaHRiRk5qYUdWdFlRdFliV3hFYVdabVIzSmhiUUVCQWdBQUFBWURBQUFBNWhZOFAzaHRiQ0IyWlhKemFXOXVQU0l4TGpBaUlHVnVZMjlrYVc1blBTSjFkR1l0TVRZaVB6NE5Dang0Y3pwelkyaGxiV0VnYVdROUlrNWxkMFJoZEdGVFpYUWlJSGh0Ykc1elBTSWlJSGh0Ykc1ek9uaHpQU0pvZEhSd09pOHZkM2QzTG5jekxtOXlaeTh5TURBeEwxaE5URk5qYUdWdFlTSWdlRzFzYm5NNmJYTmtZWFJoUFNKMWNtNDZjMk5vWlcxaGN5MXRhV055YjNOdlpuUXRZMjl0T25odGJDMXRjMlJoZEdFaVBnMEtJQ0E4ZUhNNlpXeGxiV1Z1ZENCdVlXMWxQU0pVWVdKc1pTSStEUW9nSUNBZ1BIaHpPbU52YlhCc1pYaFVlWEJsUGcwS0lDQWdJQ0FnUEhoek9uTmxjWFZsYm1ObFBnMEtJQ0FnSUNBZ0lDQThlSE02Wld4bGJXVnVkQ0J1WVcxbFBTSkxRMFJOSWlCMGVYQmxQU0o0Y3pwemRISnBibWNpSUcxelpHRjBZVHAwWVhKblpYUk9ZVzFsYzNCaFkyVTlJaUlnYldsdVQyTmpkWEp6UFNJd0lpQXZQZzBLSUNBZ0lDQWdJQ0E4ZUhNNlpXeGxiV1Z1ZENCdVlXMWxQU0pMUTAxRElpQjBlWEJsUFNKNGN6cHpkSEpwYm1jaUlHMXpaR0YwWVRwMFlYSm5aWFJPWVcxbGMzQmhZMlU5SWlJZ2JXbHVUMk5qZFhKelBTSXdJaUF2UGcwS0lDQWdJQ0FnSUNBOGVITTZaV3hsYldWdWRDQnVZVzFsUFNKS1UxcEhTQ0lnZEhsd1pUMGllSE02YzNSeWFXNW5JaUJ0YzJSaGRHRTZkR0Z5WjJWMFRtRnRaWE53WVdObFBTSWlJRzFwYms5alkzVnljejBpTUNJZ0x6NE5DaUFnSUNBZ0lDQWdQSGh6T21Wc1pXMWxiblFnYm1GdFpUMGlTbE5ZVFNJZ2RIbHdaVDBpZUhNNmMzUnlhVzVuSWlCdGMyUmhkR0U2ZEdGeVoyVjBUbUZ0WlhOd1lXTmxQU0lpSUcxcGJrOWpZM1Z5Y3owaU1DSWdMejROQ2lBZ0lDQWdJQ0FnUEhoek9tVnNaVzFsYm5RZ2JtRnRaVDBpVTB0VFNpSWdkSGx3WlQwaWVITTZjM1J5YVc1bklpQnRjMlJoZEdFNmRHRnlaMlYwVG1GdFpYTndZV05sUFNJaUlHMXBiazlqWTNWeWN6MGlNQ0lnTHo0TkNpQWdJQ0FnSUNBZ1BIaHpPbVZzWlcxbGJuUWdibUZ0WlQwaVUwdEVSQ0lnZEhsd1pUMGllSE02YzNSeWFXNW5JaUJ0YzJSaGRHRTZkR0Z5WjJWMFRtRnRaWE53WVdObFBTSWlJRzFwYms5alkzVnljejBpTUNJZ0x6NE5DaUFnSUNBZ0lDQWdQSGh6T21Wc1pXMWxiblFnYm1GdFpUMGlXRVlpSUhSNWNHVTlJbmh6T25OMGNtbHVaeUlnYlhOa1lYUmhPblJoY21kbGRFNWhiV1Z6Y0dGalpUMGlJaUJ0YVc1UFkyTjFjbk05SWpBaUlDOCtEUW9nSUNBZ0lDQWdJRHg0Y3pwbGJHVnRaVzUwSUc1aGJXVTlJbHBZVXlJZ2RIbHdaVDBpZUhNNmMzUnlhVzVuSWlCdGMyUmhkR0U2ZEdGeVoyVjBUbUZ0WlhOd1lXTmxQU0lpSUcxcGJrOWpZM1Z5Y3owaU1DSWdMejROQ2lBZ0lDQWdJQ0FnUEhoek9tVnNaVzFsYm5RZ2JtRnRaVDBpVVZOS1Uxb2lJSFI1Y0dVOUluaHpPbk4wY21sdVp5SWdiWE5rWVhSaE9uUmhjbWRsZEU1aGJXVnpjR0ZqWlQwaUlpQnRhVzVQWTJOMWNuTTlJakFpSUM4K0RRb2dJQ0FnSUNBZ0lEeDRjenBsYkdWdFpXNTBJRzVoYldVOUlsaExTMGdpSUhSNWNHVTlJbmh6T25OMGNtbHVaeUlnYlhOa1lYUmhPblJoY21kbGRFNWhiV1Z6Y0dGalpUMGlJaUJ0YVc1UFkyTjFjbk05SWpBaUlDOCtEUW9nSUNBZ0lDQWdJRHg0Y3pwbGJHVnRaVzUwSUc1aGJXVTlJbGhSUWxNaUlIUjVjR1U5SW5oek9uTjBjbWx1WnlJZ2JYTmtZWFJoT25SaGNtZGxkRTVoYldWemNHRmpaVDBpSWlCdGFXNVBZMk4xY25NOUlqQWlJQzgrRFFvZ0lDQWdJQ0FnSUR4NGN6cGxiR1Z0Wlc1MElHNWhiV1U5SWt0RFIxTWlJSFI1Y0dVOUluaHpPbk4wY21sdVp5SWdiWE5rWVhSaE9uUmhjbWRsZEU1aGJXVnpjR0ZqWlQwaUlpQnRhVzVQWTJOMWNuTTlJakFpSUM4K0RRb2dJQ0FnSUNBZ0lEeDRjenBsYkdWdFpXNTBJRzVoYldVOUlrdERXRm9pSUhSNWNHVTlJbmh6T25OMGNtbHVaeUlnYlhOa1lYUmhPblJoY21kbGRFNWhiV1Z6Y0dGalpUMGlJaUJ0YVc1UFkyTjFjbk05SWpBaUlDOCtEUW9nSUNBZ0lDQWdJRHg0Y3pwbGJHVnRaVzUwSUc1aGJXVTlJbEpUSWlCMGVYQmxQU0o0Y3pwa1pXTnBiV0ZzSWlCdGMyUmhkR0U2ZEdGeVoyVjBUbUZ0WlhOd1lXTmxQU0lpSUcxcGJrOWpZM1Z5Y3owaU1DSWdMejROQ2lBZ0lDQWdJQ0FnUEhoek9tVnNaVzFsYm5RZ2JtRnRaVDBpV1ZoU1V5SWdkSGx3WlQwaWVITTZaR1ZqYVcxaGJDSWdiWE5rWVhSaE9uUmhjbWRsZEU1aGJXVnpjR0ZqWlQwaUlpQnRhVzVQWTJOMWNuTTlJakFpSUM4K0RRb2dJQ0FnSUNBZ0lEeDRjenBsYkdWdFpXNTBJRzVoYldVOUlsbE1JaUIwZVhCbFBTSjRjenB6ZEhKcGJtY2lJRzF6WkdGMFlUcDBZWEpuWlhST1lXMWxjM0JoWTJVOUlpSWdiV2x1VDJOamRYSnpQU0l3SWlBdlBnMEtJQ0FnSUNBZ0lDQThlSE02Wld4bGJXVnVkQ0J1WVcxbFBTSkNXaUlnZEhsd1pUMGllSE02YzNSeWFXNW5JaUJ0YzJSaGRHRTZkR0Z5WjJWMFRtRnRaWE53WVdObFBTSWlJRzFwYms5alkzVnljejBpTUNJZ0x6NE5DaUFnSUNBZ0lDQWdQSGh6T21Wc1pXMWxiblFnYm1GdFpUMGlUVXRhU0NJZ2RIbHdaVDBpZUhNNmMzUnlhVzVuSWlCdGMyUmhkR0U2ZEdGeVoyVjBUbUZ0WlhOd1lXTmxQU0lpSUcxcGJrOWpZM1Z5Y3owaU1DSWdMejROQ2lBZ0lDQWdJQ0FnUEhoek9tVnNaVzFsYm5RZ2JtRnRaVDBpUzB0WVdTSWdkSGx3WlQwaWVITTZjM1J5YVc1bklpQnRjMlJoZEdFNmRHRnlaMlYwVG1GdFpYTndZV05sUFNJaUlHMXBiazlqWTNWeWN6MGlNQ0lnTHo0TkNpQWdJQ0FnSUNBZ1BIaHpPbVZzWlcxbGJuUWdibUZ0WlQwaVMxTlRTaUlnZEhsd1pUMGllSE02YzNSeWFXNW5JaUJ0YzJSaGRHRTZkR0Z5WjJWMFRtRnRaWE53WVdObFBTSWlJRzFwYms5alkzVnljejBpTUNJZ0x6NE5DaUFnSUNBZ0lDQWdQSGh6T21Wc1pXMWxiblFnYm1GdFpUMGlVMFpDV0NJZ2RIbHdaVDBpZUhNNmMzUnlhVzVuSWlCdGMyUmhkR0U2ZEdGeVoyVjBUbUZ0WlhOd1lXTmxQU0lpSUcxcGJrOWpZM1Z5Y3owaU1DSWdMejROQ2lBZ0lDQWdJQ0FnUEhoek9tVnNaVzFsYm5RZ2JtRnRaVDBpUzFOWVV5SWdkSGx3WlQwaWVITTZjM1J5YVc1bklpQnRjMlJoZEdFNmRHRnlaMlYwVG1GdFpYTndZV05sUFNJaUlHMXBiazlqWTNWeWN6MGlNQ0lnTHo0TkNpQWdJQ0FnSUNBZ1BIaHpPbVZzWlcxbGJuUWdibUZ0WlQwaVUxRlRUU0lnZEhsd1pUMGllSE02YzNSeWFXNW5JaUJ0YzJSaGRHRTZkR0Z5WjJWMFRtRnRaWE53WVdObFBTSWlJRzFwYms5alkzVnljejBpTUNJZ0x6NE5DaUFnSUNBZ0lDQWdQSGh6T21Wc1pXMWxiblFnYm1GdFpUMGlSa3BZV2lJZ2RIbHdaVDBpZUhNNmMzUnlhVzVuSWlCdGMyUmhkR0U2ZEdGeVoyVjBUbUZ0WlhOd1lXTmxQU0lpSUcxcGJrOWpZM1Z5Y3owaU1DSWdMejROQ2lBZ0lDQWdJQ0FnUEhoek9tVnNaVzFsYm5RZ2JtRnRaVDBpU2tOT1VpSWdkSGx3WlQwaWVITTZjM1J5YVc1bklpQnRjMlJoZEdFNmRHRnlaMlYwVG1GdFpYTndZV05sUFNJaUlHMXBiazlqWTNWeWN6MGlNQ0lnTHo0TkNpQWdJQ0FnSUNBZ1BIaHpPbVZzWlcxbGJuUWdibUZ0WlQwaVVrNGlJSFI1Y0dVOUluaHpPbVJsWTJsdFlXd2lJRzF6WkdGMFlUcDBZWEpuWlhST1lXMWxjM0JoWTJVOUlpSWdiV2x1VDJOamRYSnpQU0l3SWlBdlBnMEtJQ0FnSUNBZ1BDOTRjenB6WlhGMVpXNWpaVDROQ2lBZ0lDQThMM2h6T21OdmJYQnNaWGhVZVhCbFBnMEtJQ0E4TDNoek9tVnNaVzFsYm5RK0RRb2dJRHg0Y3pwbGJHVnRaVzUwSUc1aGJXVTlJazVsZDBSaGRHRlRaWFFpSUcxelpHRjBZVHBKYzBSaGRHRlRaWFE5SW5SeWRXVWlJRzF6WkdGMFlUcE1iMk5oYkdVOUlucG9MVU5PSWo0TkNpQWdJQ0E4ZUhNNlkyOXRjR3hsZUZSNWNHVStEUW9nSUNBZ0lDQThlSE02WTJodmFXTmxJRzFoZUU5alkzVnljejBpZFc1aWIzVnVaR1ZrSWlBdlBnMEtJQ0FnSUR3dmVITTZZMjl0Y0d4bGVGUjVjR1UrRFFvZ0lEd3ZlSE02Wld4bGJXVnVkRDROQ2p3dmVITTZjMk5vWlcxaFBnWUVBQUFBclFZOFpHbG1abWR5T21ScFptWm5jbUZ0SUhodGJHNXpPbTF6WkdGMFlUMGlkWEp1T25OamFHVnRZWE10YldsamNtOXpiMlowTFdOdmJUcDRiV3d0YlhOa1lYUmhJaUI0Yld4dWN6cGthV1ptWjNJOUluVnlianB6WTJobGJXRnpMVzFwWTNKdmMyOW1kQzFqYjIwNmVHMXNMV1JwWm1abmNtRnRMWFl4SWo0TkNpQWdQRTVsZDBSaGRHRlRaWFErRFFvZ0lDQWdQRlJoWW14bElHUnBabVpuY2pwcFpEMGlWR0ZpYkdVeElpQnRjMlJoZEdFNmNtOTNUM0prWlhJOUlqQWlQZzBLSUNBZ0lDQWdQRXREUkUwK1UwdFlNRE16TXpROEwwdERSRTArRFFvZ0lDQWdJQ0E4UzBOTlF6N2t2NkhtZ2EvbW80RG50S0x2dklqbnZaSG51NXpvcjc3bnFJdnZ2SWs4TDB0RFRVTStEUW9nSUNBZ0lDQThTbE5hUjBnK09UQXhOamc4TDBwVFdrZElQZzBLSUNBZ0lDQWdQRXBUV0UwKzVaQzA1TG0rNXJpRlBDOUtVMWhOUGcwS0lDQWdJQ0FnUEZOTFUwbys1WkdvNUxpSjU2eXNPU3d4TU9pS2dudm5yS3cxTFRFNDVaR29mVHd2VTB0VFNqNE5DaUFnSUNBZ0lEeFRTMFJFUHVlVXRlV3RrT21ZaGVpbmlPV3VwRHd2VTB0RVJENE5DaUFnSUNBZ0lEeFlSajR5TGpBOEwxaEdQZzBLSUNBZ0lDQWdQRnBZVXo0eUxqQXRNQzR3UEM5YVdGTStEUW9nSUNBZ0lDQThVVk5LVTFvK01EVXRNVGc4TDFGVFNsTmFQZzBLSUNBZ0lDQWdQRmhMUzBnK0tESXdNakl0TWpBeU15MHlLUzFUUzFnd016TXpOQzA1TURFMk9DMHhQQzlZUzB0SVBnMEtJQ0FnSUNBZ1BGaFJRbE0rNXB5czZZT29QQzlZVVVKVFBnMEtJQ0FnSUNBZ1BFdERSMU0rNUxxNjVwYUg1YTJtNTZlUlF1ZXh1K21BaWVTL3Jqd3ZTME5IVXo0TkNpQWdJQ0FnSUR4TFExaGFQdVdGck9XRnNlbUFpZVMvcnVpdnZqd3ZTME5ZV2o0TkNpQWdJQ0FnSUR4U1V6NDBOVHd2VWxNK0RRb2dJQ0FnSUNBOFdWaFNVejQwTkR3dldWaFNVejROQ2lBZ0lDQWdJRHhMUzFoWlB1YVZtZVdLb2VXa2hEd3ZTMHRZV1Q0TkNpQWdJQ0FnSUR4S1EwNVNQbng4ZkR3dlNrTk9VajROQ2lBZ0lDQWdJRHhTVGo0eFBDOVNUajROQ2lBZ0lDQThMMVJoWW14bFBnMEtJQ0E4TDA1bGQwUmhkR0ZUWlhRK0RRbzhMMlJwWm1abmNqcGthV1ptWjNKaGJUNEw+Oz4+Oz47bDxpPDE+Oz47bDx0PDtsPGk8MT47aTwzPjtpPDU+O2k8Nz47aTwxMT47aTwxMz47aTwxNT47aTwxNz47PjtsPHQ8cDxwPGw8VGV4dDs+O2w8MTs+Pjs+Ozs+O3Q8cDxwPGw8VGV4dDs+O2w8MTs+Pjs+Ozs+O3Q8cDxwPGw8VGV4dDs+O2w8MTs+Pjs+Ozs+O3Q8cDxwPGw8VGV4dDs+O2w8MTs+Pjs+Ozs+O3Q8cDxwPGw8RW5hYmxlZDs+O2w8bzxmPjs+Pjs+Ozs+O3Q8cDxwPGw8RW5hYmxlZDs+O2w8bzxmPjs+Pjs+Ozs+O3Q8cDxwPGw8RW5hYmxlZDs+O2w8bzxmPjs+Pjs+Ozs+O3Q8cDxwPGw8RW5hYmxlZDs+O2w8bzxmPjs+Pjs+Ozs+Oz4+Oz4+O3Q8QDA8cDxwPGw8UGFnZUNvdW50O18hSXRlbUNvdW50O18hRGF0YVNvdXJjZUl0ZW1Db3VudDtEYXRhS2V5czs+O2w8aTwxPjtpPDA+O2k8MD47bDw+Oz4+Oz47Ozs7Ozs7Ozs7Pjs7Pjt0PEAwPDs7Ozs7Ozs7Ozs+Ozs+O3Q8O2w8aTwzPjs+O2w8dDxAMDw7Ozs7Ozs7Oz47Oz47Pj47Pj47Pj47bDxrY21jR3JpZDpfY3RsMjp4aztrY21jR3JpZDpfY3RsMjpqYzs+Pr8Vfo9DuEmfZgkVXO/B612J5dUi",
        "__VIEWSTATEGENERATOR": "03DFB912",
        "ddl_kcxz": "",
        "ddl_ywyl": "有".encode('gb2312'),
        "ddl_kcgs": "",
        "ddl_xqbs": "1",
        "ddl_sksj": "",
        "TextBox1": "",
        "kcmcGrid:_ctl2:xk": "on",
        "kcmcGrid:_ctl2:jcnr": "|||",
        "dpkcmcGrid:txtChoosePage": "1",
        "dpkcmcGrid:txtPageSize": "15",
        "Button1": "++%CC%E1%BD%BB++"
    }
    url = "http://jiaowu.hncst.edu.cn/xf_xsqxxxk.aspx?xh={0}".format(number)
    header = {
        "Referer": "http://jiaowu.hncst.edu.cn/xs_main.aspx?xh={0}".format(number),
        "Cookie": "ASP.NET_SessionId={0}".format(cookie),
    }
    for x in range(1, 100):
        response = requests.post(url, headers=header, data=data)
        print(len(response.text))
        soup = BS(response.text, 'html.parser')
        table_html = str(soup.find("table", id="DataGrid2"))
        if "return confirm(" in table_html:
            print("----------------------抢课成功-----------------------")
            qiangke1()
            break
number = input("请输入学号: ")
cookie = input("请输入登录后的Cookie: ")
while 1:
    code = input("抢课【2】，验证抢到的课【1】:")
    if str.isdigit(code):
        if code == '1':
            qiangke1()
        elif code == '2':
            qiangke2()
        else:
            print("没有这个模式")
            exit()
    else:
        print("输入的不是数字")


