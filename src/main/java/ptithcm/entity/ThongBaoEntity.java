package ptithcm.entity;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.Table;
import java.util.Date;

@Entity
@Table(name = "THONGBAO")
public class ThongBaoEntity {
    @Id
    @GeneratedValue
    @Column(name="MATB")
    private int maTB;
    
    @ManyToOne
    @JoinColumn(name="NGUOINHAN")
    private NguoiDungEntity nguoiDung;
    
    @Column(name = "NOIDUNG")
    private String noiDung;

    @Column(name = "NOIDUNG_LINKED")
    private String noiDungLinked;
    
    @Column(name = "IS_READ")
    private boolean isRead;
    
    @Column(name = "THOIGIANTAO", nullable = false, columnDefinition = "DATETIME DEFAULT GETDATE()")
    private Date thoiGianTao;

    // Getter và setter cho maTB
    public int getMaTB() {
        return maTB;
    }

    public void setMaTB(int maTB) {
        this.maTB = maTB;
    }

    // Getter và setter cho nguoiDung
    public NguoiDungEntity getNguoiDung() {
        return nguoiDung;
    }

    public void setNguoiDung(NguoiDungEntity nguoiDung) {
        this.nguoiDung = nguoiDung;
    }

    // Getter và setter cho noiDung
    public String getNoiDung() {
        return noiDung;
    }

    public void setNoiDung(String noiDung) {
        this.noiDung = noiDung;
    }

    // Getter và setter cho noiDungLinked
    public String getNoiDungLinked() {
        return noiDungLinked;
    }

    public void setNoiDungLinked(String noiDungLinked) {
        this.noiDungLinked = noiDungLinked;
    }

    // Getter và setter cho isRead
    public boolean isRead() {
        return isRead;
    }

    public void setRead(boolean isRead) {
        this.isRead = isRead;
    }

    // Getter và setter cho thoiGianTao
    public Date getThoiGianTao() {
        return thoiGianTao;
    }

    public void setThoiGianTao(Date thoiGianTao) {
        this.thoiGianTao = thoiGianTao;
    }
}